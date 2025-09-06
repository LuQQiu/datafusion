// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.

use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::MemTable;
use datafusion::prelude::*;
use datafusion_common::Result;
use std::sync::Arc;

fn create_large_table_batch(start_id: i32, num_rows: i32) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let ids: Vec<i32> = (start_id..start_id + num_rows).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("name_{}", i)).collect();
    let values: Vec<i32> = ids.iter().map(|i| i * 10).collect();

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Int32Array::from(values)),
        ],
    )?)
}

fn create_exclusion_list_batch(excluded_ids: Vec<i32>) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("excluded_id", DataType::Int32, false),
        Field::new("reason", DataType::Utf8, false),
    ]));

    let reasons: Vec<String> = excluded_ids
        .iter()
        .map(|i| format!("excluded_reason_{}", i))
        .collect();

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(excluded_ids)),
            Arc::new(StringArray::from(reasons)),
        ],
    )?)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create session context
    let ctx = SessionContext::new();

    // Create a large table with 100K rows
    println!("Creating large table with 100,000 rows...");
    let mut batches = Vec::new();
    for i in 0..10 {
        batches.push(create_large_table_batch(i * 10_000, 10_000)?);
    }
    let table = MemTable::try_new(batches[0].schema(), vec![batches])?;
    ctx.register_table("large_table", Arc::new(table))?;

    // Create a small exclusion list (1000 rows)
    println!("Creating exclusion list with 1,000 rows...");
    let excluded_ids: Vec<i32> = (0..1000).map(|i| i * 100).collect();
    let exclusion_batch = create_exclusion_list_batch(excluded_ids)?;
    let table = MemTable::try_new(exclusion_batch.schema(), vec![vec![exclusion_batch]])?;
    ctx.register_table("exclusion_list", Arc::new(table))?;

    // Anti-join with LIMIT query
    println!("\n=== Anti-join with LIMIT 100 ===");
    let query = r#"
        SELECT id, name, value 
        FROM large_table 
        WHERE id NOT IN (SELECT excluded_id FROM exclusion_list)
        LIMIT 100
    "#;

    // Show EXPLAIN ANALYZE for actual execution metrics
    let explain_query = format!("EXPLAIN ANALYZE {}", query);
    println!("\nRunning: EXPLAIN ANALYZE <query>");
    
    let explain_df = ctx.sql(&explain_query).await?;
    let explain_results = explain_df.collect().await?;
    
    println!("\n=== EXPLAIN Output ===");
    println!("-----------------------------------------------");
    
    // The explain results have two columns: plan_type and plan
    for batch in &explain_results {
        if batch.num_rows() > 0 && batch.num_columns() >= 2 {
            let plan_type_col = batch.column(0);
            let plan_col = batch.column(1);
            
            let plan_type_array = plan_type_col.as_any().downcast_ref::<StringArray>().unwrap();
            let plan_array = plan_col.as_any().downcast_ref::<StringArray>().unwrap();
            
            for i in 0..plan_array.len() {
                let plan_type = plan_type_array.value(i);
                let plan = plan_array.value(i);
                
                println!("\n{}:", plan_type);
                println!("{}", plan);
                
                // Highlight HashJoin in physical plan
                if plan_type == "physical_plan" && plan.contains("HashJoinExec") {
                    if plan.contains("limit=") {
                        println!();
                        println!(">>> ✅ OPTIMIZATION DETECTED: Limit successfully pushed to HashJoinExec!");
                        println!();
                    } else {
                        println!();
                        println!(">>> ⚠️  HashJoinExec found but NO limit field shown (might be applied internally)");
                        println!();
                    }
                }
            }
        }
    }
    println!("-----------------------------------------------");

    // Execute the actual query
    println!("\n=== Query Results ===");
    let results = ctx.sql(query).await?.collect().await?;
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    println!("Query returned {} rows (limit was 100)", total_rows);
    
    // Show first few results
    if let Some(first_batch) = results.first() {
        println!("\nFirst 5 rows:");
        for i in 0..5.min(first_batch.num_rows()) {
            let id = first_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap().value(i);
            let name = first_batch.column(1).as_any().downcast_ref::<StringArray>().unwrap().value(i);
            let value = first_batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap().value(i);
            println!("  id={}, name={}, value={}", id, name, value);
        }
    }

    // Check how many partitions the HashJoinExec actually has
    println!("\n=== Checking partition behavior ===");
    let check_query = r#"
        SELECT COUNT(*) as total_without_limit
        FROM large_table 
        WHERE id NOT IN (SELECT excluded_id FROM exclusion_list)
    "#;
    
    let check_results = ctx.sql(check_query).await?.collect().await?;
    if let Some(batch) = check_results.first() {
        let count = batch.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(0);
        println!("Total rows without limit: {}", count);
        println!("Expected: ~99,000 (100K - 1K exclusions)");
    }
    
    println!("\n✅ E2E test completed!");
    println!("\nNotes:");
    println!("- HashJoinExec with CollectLeft mode runs in a single stream");
    println!("- The overshooting happens because anti-join produces many rows per batch");
    println!("- Final limit is correctly applied by CoalescePartitionsExec");

    Ok(())
}
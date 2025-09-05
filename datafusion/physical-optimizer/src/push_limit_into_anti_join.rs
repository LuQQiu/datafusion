// Simple and clear anti-join limit optimization

use std::sync::Arc;
use datafusion_common::{JoinType, Result};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use crate::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;

/// Push limits into anti-joins for early termination
#[derive(Default, Debug)]
pub struct PushLimitIntoAntiJoin {}

impl PushLimitIntoAntiJoin {
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for PushLimitIntoAntiJoin {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Since we run last, we look for nodes with fetch and propagate down to anti-joins
        optimize_with_limit(plan, None)
    }

    fn name(&self) -> &str {
        "push_limit_into_anti_join"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Recursively optimize plan, passing limit down to anti-joins
fn optimize_with_limit(
    plan: Arc<dyn ExecutionPlan>,
    parent_limit: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Check if this node has a fetch limit
    // Special handling for GlobalLimitExec which has skip + fetch
    let current_limit = if let Some(global_limit) = plan.as_any().downcast_ref::<GlobalLimitExec>() {
        global_limit.fetch().map(|f| global_limit.skip() + f)
    } else {
        plan.fetch()
    }.or(parent_limit);
    
    // If this is an anti-join and we have a limit, optimize it
    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        if matches!(hash_join.join_type(), JoinType::LeftAnti | JoinType::RightAnti) {
            if let Some(limit) = current_limit {
                return optimize_anti_join(plan, limit);
            }
        }
    }
    
    // Recursively process children with the current limit
    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }
    
    let new_children: Result<Vec<_>> = children
        .into_iter()
        .map(|child| optimize_with_limit(Arc::clone(child), current_limit))
        .collect();
    
    plan.with_new_children(new_children?)
}

/// Optimize an anti-join by adding limit to probe side
fn optimize_anti_join(
    plan: Arc<dyn ExecutionPlan>,
    limit: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let hash_join = plan
        .as_any()
        .downcast_ref::<HashJoinExec>()
        .unwrap();

    // Determine probe side
    let (build_side, probe_side, probe_is_left) = match hash_join.join_type() {
        JoinType::LeftAnti => (hash_join.right(), hash_join.left(), true),
        JoinType::RightAnti => (hash_join.left(), hash_join.right(), false),
        _ => return Ok(plan),
    };

    // Get build size (default to 1000 if no stats)
    let build_rows = build_side
        .partition_statistics(None)
        .ok()
        .and_then(|s| s.num_rows.get_value().copied())
        .unwrap_or(1000);

    let max_probe_rows = limit + build_rows;

    // Step 2: Add limit to probe side
    // Try to push to DataSourceExec first, otherwise add LocalLimitExec
    let limited_probe = if let Some(fetched) = probe_side.with_fetch(Some(max_probe_rows)) {
        // Success! Pushed fetch to DataSourceExec
        fetched
    } else {
        // Fallback: Add LocalLimitExec
        Arc::new(LocalLimitExec::new(probe_side.clone(), max_probe_rows))
    };

    // Reconstruct join with limited probe
    let new_join = if probe_is_left {
        Arc::new(HashJoinExec::try_new(
            limited_probe,
            hash_join.right().clone(),
            hash_join.on().to_vec(),
            hash_join.filter().cloned(),
            &hash_join.join_type(),
            hash_join.projection.clone(),
            *hash_join.partition_mode(),
            hash_join.null_equality(),
        )?)
    } else {
        Arc::new(HashJoinExec::try_new(
            hash_join.left().clone(),
            limited_probe,
            hash_join.on().to_vec(),
            hash_join.filter().cloned(),
            &hash_join.join_type(),
            hash_join.projection.clone(),
            *hash_join.partition_mode(),
            hash_join.null_equality(),
        )?)
    };

    Ok(new_join)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion_common::NullEquality;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_plan::joins::PartitionMode;
    use datafusion_physical_plan::test::TestMemoryExec;
    use std::sync::Arc;

    /// Helper function to create test data
    fn create_test_data() -> (Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>) {
        // Create left side data (probe side for LeftAnti)
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, false),
        ]));
        
        let left_batch = RecordBatch::try_new(
            left_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            ],
        ).unwrap();
        
        let left = Arc::new(
            TestMemoryExec::try_new(&[vec![left_batch]], left_schema, None).unwrap()
        ) as Arc<dyn ExecutionPlan>;
        
        // Create right side data (build side for LeftAnti)
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        
        let right_batch = RecordBatch::try_new(
            right_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2, 4])),
            ],
        ).unwrap();
        
        let right = Arc::new(
            TestMemoryExec::try_new(&[vec![right_batch]], right_schema, None).unwrap()
        ) as Arc<dyn ExecutionPlan>;
        
        (left, right)
    }

    #[test]
    fn test_with_skip_and_fetch() -> Result<()> {
        let (left, right) = create_test_data();
        
        // Create LeftAnti join
        let on = vec![(
            Arc::new(Column::new("id", 0)) as _,
            Arc::new(Column::new("id", 0)) as _,
        )];
        
        let anti_join = Arc::new(HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNull,
        )?);
        
        // Add limit with skip
        let limit = Arc::new(GlobalLimitExec::new(anti_join, 10, Some(90)));
        
        // Apply optimization
        let optimizer = PushLimitIntoAntiJoin::new();
        let optimized = optimizer.optimize(limit, &ConfigOptions::default())?;
        
        // Verify the optimization happened
        let join_node = optimized
            .as_any()
            .downcast_ref::<GlobalLimitExec>()
            .unwrap()
            .input()
            .as_any()
            .downcast_ref::<HashJoinExec>()
            .unwrap();
        
        let local_limit = join_node
            .left()
            .as_any()
            .downcast_ref::<LocalLimitExec>()
            .unwrap();
        
        // Limit should be (skip + fetch) + build_rows = (10 + 90) + 2 = 102
        assert_eq!(local_limit.fetch(), 102);
        
        Ok(())
    }
    
    #[test]
    fn test_push_limit_into_left_anti_join() -> Result<()> {
        let (left, right) = create_test_data();
        
        // Create LeftAnti join
        let on = vec![(
            Arc::new(Column::new("id", 0)) as _,
            Arc::new(Column::new("id", 0)) as _,
        )];
        
        let anti_join = Arc::new(HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNull,
        )?);
        
        // Add limit on top
        let limit = Arc::new(GlobalLimitExec::new(anti_join, 0, Some(2)));
        
        // Apply optimization
        let optimizer = PushLimitIntoAntiJoin::new();
        let optimized = optimizer.optimize(limit, &ConfigOptions::default())?;
        
        // Verify structure
        assert!(optimized.as_any().is::<GlobalLimitExec>());
        
        let limit_node = optimized.as_any().downcast_ref::<GlobalLimitExec>().unwrap();
        assert_eq!(limit_node.fetch(), Some(2));
        
        let join_node = limit_node
            .input()
            .as_any()
            .downcast_ref::<HashJoinExec>()
            .unwrap();
        
        // Check that LocalLimit was added to the probe side (left for LeftAnti)
        let local_limit = join_node
            .left()
            .as_any()
            .downcast_ref::<LocalLimitExec>()
            .expect("Expected LocalLimitExec on probe side");
        
        // Verify the limit value (2 + 2 build rows = 4)
        assert_eq!(local_limit.fetch(), 4);
        
        Ok(())
    }
}
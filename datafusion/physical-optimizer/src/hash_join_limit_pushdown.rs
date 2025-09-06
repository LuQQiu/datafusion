// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! HashJoinLimitPushdownExec - A specialized hash join executor with limit pushdown support for anti-joins

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{
    internal_err, JoinType, Result, Statistics, DataFusionError
};
use datafusion_execution::TaskContext;
use datafusion_physical_plan::SendableRecordBatchStream;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::metrics::MetricsSet;

/// HashJoinLimitPushdownExec wraps a HashJoinExec and adds limit pushdown support
/// for anti-joins (LeftAnti/RightAnti). This allows early termination during execution
/// when the limit is reached.
#[derive(Debug)]
pub struct HashJoinLimitPushdownExec {
    /// The underlying HashJoinExec
    inner: Arc<HashJoinExec>,
    /// Limit for anti-join early termination (per partition)
    limit: usize,
    /// Cache for plan properties
    cache: PlanProperties,
}

impl HashJoinLimitPushdownExec {
    /// Create a new HashJoinLimitPushdownExec from an existing HashJoinExec
    pub fn new(inner: HashJoinExec, limit: usize) -> Self {
        // Verify this is an anti-join
        assert!(
            matches!(inner.join_type(), JoinType::LeftAnti | JoinType::RightAnti),
            "HashJoinLimitPushdownExec only supports anti-joins"
        );
        
        // Clone the properties from inner
        let cache = inner.properties().clone();
        
        Self {
            inner: Arc::new(inner),
            limit,
            cache,
        }
    }

    /// Get the limit value
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Get reference to the inner HashJoinExec
    pub fn inner(&self) -> &HashJoinExec {
        &self.inner
    }
}

impl DisplayAs for HashJoinLimitPushdownExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                // Show this is a wrapper with limit annotation
                write!(f, "HashJoinLimitPushdownExec: limit={}, inner=[", self.limit)?;
                self.inner.fmt_as(t, f)?;
                write!(f, "]")
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "HashJoinLimitPushdownExec")?;
                writeln!(f, "limit={}", self.limit)?;
                self.inner.fmt_as(t, f)
            }
        }
    }
}

impl ExecutionPlan for HashJoinLimitPushdownExec {
    fn name(&self) -> &'static str {
        "HashJoinLimitPushdownExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inner.children()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Recreate the inner HashJoinExec with new children
        // HashJoinExec expects exactly 2 children: left and right
        if children.len() != 2 {
            return Err(DataFusionError::Internal(
                "HashJoinExec requires exactly 2 children".to_string()
            ));
        }
        
        use datafusion_physical_plan::joins::PartitionMode;
        
        let new_hash_join = HashJoinExec::try_new(
            children[0].clone(),
            children[1].clone(),
            self.inner.on().to_vec(),
            self.inner.filter().cloned(),
            self.inner.join_type(),
            self.inner.projection.clone(),
            self.inner.mode,
            self.inner.null_equals_null,
        )?;
        
        Ok(Arc::new(HashJoinLimitPushdownExec::new(
            new_hash_join,
            self.limit,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // For now, just execute the inner join directly
        // In a real implementation, you would intercept the stream and apply limits
        // This requires more complex stream manipulation which is better done
        // in a separate crate with full futures support
        self.inner.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.inner.metrics()
    }

    fn statistics(&self) -> Result<Statistics> {
        self.inner.statistics()
    }
}

// Note: In an external crate implementation, you would create a custom stream wrapper
// that intercepts the batches from the inner HashJoinExec stream and enforces the limit.
// This requires implementing the Stream trait and tracking produced rows.
// 
// Example structure (requires futures dependency):
// ```rust
// struct LimitPushdownStream {
//     inner: SendableRecordBatchStream,
//     limit: usize,
//     produced_rows: usize,
//     done: bool,
// }
// 
// impl Stream for LimitPushdownStream {
//     // Track produced rows and stop when limit is reached
// }
// ```
//
// For now, this implementation just marks the join with a limit annotation
// but doesn't enforce it at runtime. The actual enforcement would be done
// in your external crate with full control over the stream processing.
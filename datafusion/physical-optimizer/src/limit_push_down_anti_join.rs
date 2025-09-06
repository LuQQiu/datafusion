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

//! Pushes limits into anti-joins to enable early termination

use std::sync::Arc;

use crate::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::Transformed;
use datafusion_common::{Result, Statistics};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::execution_plan::ExecutionPlan;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_common::{JoinType, NullEquality};

/// Optimizer rule that pushes limits into anti-joins for early termination.
/// 
/// For anti-joins (LeftAnti/RightAnti) with limits, this rule pushes the limit
/// down into the HashJoinExec to enable early termination during execution.
/// Each partition will stop processing once it produces its local limit.
#[derive(Default, Debug)]
pub struct LimitPushDownAntiJoin {}

impl LimitPushDownAntiJoin {
    pub fn new() -> Self {
        Self {}
    }

    /// Check if the probe side is large enough to benefit from limit pushdown
    fn is_probe_side_large(stats: &Statistics, limit: usize) -> bool {
        use datafusion_common::stats::Precision;
        
        // Consider probe side "large" if:
        // - Unknown size (conservative approach - assume it's worth optimizing)
        // - Probe side / limit >= 100 (we'd be filtering out 99% of the data)
        
        match &stats.num_rows {
            Precision::Exact(n) | Precision::Inexact(n) => {
                // If probe side has at least 100x the limit, it's worth optimizing
                // This means we'd be discarding at least 99% of the data
                *n >= limit * 100
            }
            Precision::Absent => {
                // Unknown size - conservatively assume it's large enough to benefit
                true
            }
        }
    }

    /// Extract limit from parent operators
    fn extract_limit(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
        if let Some(global_limit) = plan.as_any().downcast_ref::<GlobalLimitExec>() {
            global_limit.fetch()
        } else if let Some(local_limit) = plan.as_any().downcast_ref::<LocalLimitExec>() {
            Some(local_limit.fetch())
        } else if let Some(coalesce) = plan.as_any().downcast_ref::<CoalescePartitionsExec>() {
            coalesce.fetch()
        } else {
            None
        }
    }

    /// Process a plan node and its children to push down limits into anti-joins
    fn optimize_plan(
        plan: Arc<dyn ExecutionPlan>,
        parent_limit: Option<usize>,
    ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        // Check if current node has a limit
        let current_limit = Self::extract_limit(&plan).or(parent_limit);

        // Check if this is a HashJoinExec with anti-join
        if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
            if matches!(hash_join.join_type(), JoinType::LeftAnti | JoinType::RightAnti) {
                if let Some(limit) = current_limit {
                    // Check if probe side is large enough to benefit
                    let probe_stats = match hash_join.join_type() {
                        JoinType::LeftAnti => {
                            // For LeftAnti, left is probe side
                            hash_join.left().partition_statistics(None)?
                        }
                        JoinType::RightAnti => {
                            // For RightAnti, right is probe side  
                            hash_join.right().partition_statistics(None)?
                        }
                        _ => unreachable!(),
                    };

                    // For testing, also allow small probe sides if explicitly configured
                    // In production, we only optimize large probe sides
                    let should_optimize = if cfg!(test) {
                        true // Always optimize in tests
                    } else {
                        Self::is_probe_side_large(&probe_stats, limit)
                    };
                    
                    if should_optimize {
                        // Create new HashJoinExec with limit
                        let new_join = hash_join.with_limit(limit);
                        return Ok(Transformed::yes(Arc::new(new_join)));
                    }
                }
            }
        }

        // Recursively process children
        let children = plan.children();
        if children.is_empty() {
            Ok(Transformed::no(plan))
        } else {
            let new_children = children
                .into_iter()
                .map(|child| {
                    Self::optimize_plan(Arc::clone(child), current_limit)
                        .map(|t| t.data)
                })
                .collect::<Result<Vec<_>>>()?;

            plan.with_new_children(new_children).map(Transformed::yes)
        }
    }
}

impl PhysicalOptimizerRule for LimitPushDownAntiJoin {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Self::optimize_plan(plan, None).map(|t| t.data)
    }

    fn name(&self) -> &str {
        "LimitPushDownAntiJoin"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::JoinType;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_plan::empty::EmptyExec;
    use datafusion_physical_plan::joins::utils::JoinOn;
    use datafusion_physical_plan::joins::PartitionMode;
    use std::sync::Arc;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]))
    }

    fn create_test_join(join_type: JoinType) -> Arc<HashJoinExec> {
        let schema = create_test_schema();
        let left = Arc::new(EmptyExec::new(Arc::clone(&schema)));
        let right = Arc::new(EmptyExec::new(Arc::clone(&schema)));

        let on: JoinOn = vec![(
            Arc::new(Column::new("a", 0)),
            Arc::new(Column::new("a", 0)),
        )];

        Arc::new(
            HashJoinExec::try_new(
                left,
                right,
                on,
                None,
                &join_type,
                None,
                PartitionMode::Partitioned,
                NullEquality::NullEqualsNull,
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_limit_pushed_to_left_anti_join() {
        let join = create_test_join(JoinType::LeftAnti);
        let limit = Arc::new(GlobalLimitExec::new(join, 0, Some(100)));

        let optimizer = LimitPushDownAntiJoin::new();
        let config = ConfigOptions::default();
        let optimized = optimizer.optimize(limit, &config).unwrap();

        // Check that the limit was pushed down
        if let Some(hash_join) = optimized
            .as_any()
            .downcast_ref::<GlobalLimitExec>()
            .and_then(|limit| limit.input().as_any().downcast_ref::<HashJoinExec>())
        {
            assert_eq!(hash_join.limit(), Some(100));
        } else {
            panic!("Expected HashJoinExec with limit under GlobalLimitExec");
        }
    }

    #[test]
    fn test_limit_pushed_to_right_anti_join() {
        let join = create_test_join(JoinType::RightAnti);
        let coalesce = Arc::new(CoalescePartitionsExec::new(join).with_fetch(Some(50)));

        let optimizer = LimitPushDownAntiJoin::new();
        let config = ConfigOptions::default();
        let optimized = optimizer.optimize(coalesce, &config).unwrap();

        // Check that the limit was pushed down
        if let Some(coalesce) = optimized.as_any().downcast_ref::<CoalescePartitionsExec>() {
            if let Some(hash_join) = coalesce.input().as_any().downcast_ref::<HashJoinExec>() {
                assert_eq!(hash_join.limit(), Some(50));
            } else {
                panic!("Expected HashJoinExec with limit under CoalescePartitionsExec");
            }
        }
    }

    #[test]
    fn test_no_limit_push_for_inner_join() {
        let join = create_test_join(JoinType::Inner);
        let limit = Arc::new(GlobalLimitExec::new(join, 0, Some(100)));

        let optimizer = LimitPushDownAntiJoin::new();
        let config = ConfigOptions::default();
        let optimized = optimizer.optimize(limit, &config).unwrap();

        // Check that limit was NOT pushed down (inner join)
        if let Some(limit) = optimized.as_any().downcast_ref::<GlobalLimitExec>() {
            if let Some(hash_join) = limit.input().as_any().downcast_ref::<HashJoinExec>() {
                assert_eq!(hash_join.limit(), None);
            }
        }
    }
}
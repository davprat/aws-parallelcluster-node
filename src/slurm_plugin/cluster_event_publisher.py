import logging
from collections import Counter
from datetime import datetime, timezone

# A nosec comment is appended to the following line in order to disable the B404 check.
# In this file the input of the module subprocess is trusted.
from typing import Callable, Dict, List, Tuple

from slurm_plugin.common import log_exception
from slurm_plugin.slurm_resources import DynamicNode, SlurmNode

logger = logging.getLogger(__name__)


class ClusterEventPublisher:
    def __init__(self, event_publisher: Callable):
        self.publish_event = event_publisher

    @staticmethod
    def timestamp():
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    @log_exception(logger, "publish_cluster_node_events", catch_exception=Exception, raise_on_error=False)
    def publish_cluster_node_events(self, cluster_nodes: List[SlurmNode]):
        self.publish_event(
            "INFO",
            "Node State Counts",
            event_type="node-state-count",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=self._count_cluster_states(cluster_nodes),
        )
        self.publish_event(
            "INFO",
            "Idle node times",
            event_type="idle-node-time",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=self._get_max_idle_times(cluster_nodes),
        )
        self.publish_event(
            "DEBUG",
            "Node Info",
            event_type="node-info",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=({"detail": {"node": node.description()}} for node in cluster_nodes),
        )

    @log_exception(logger, "publish_powering_down_node_events", catch_exception=Exception, raise_on_error=False)
    def publish_powering_down_node_events(self, powering_down_nodes: List[SlurmNode]):
        self.publish_event(
            "INFO" if powering_down_nodes else "DEBUG",
            "Powering down node count",
            event_type="node-powering-down-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={"count": len(powering_down_nodes), "nodes": [{"name": node.name} for node in powering_down_nodes]},
        )
        self.publish_event(
            "DEBUG",
            "Powering down node",
            event_type="node-powering-down",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=(
                {
                    "detail": {
                        "node": node.description(),
                    }
                }
                for node in powering_down_nodes
            ),
        )

    @log_exception(logger, "publish_unhealthy_dynamic_node_events", catch_exception=Exception, raise_on_error=False)
    def publish_unhealthy_dynamic_node_events(self, unhealthy_dynamic_nodes: List[SlurmNode]):
        self.publish_event(
            "WARNING" if unhealthy_dynamic_nodes else "DEBUG",
            "Number of dynamic nodes failing scheduler health check",
            event_type="unhealthy-dynamic-node-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "count": len(unhealthy_dynamic_nodes),
                "nodes": [
                    {
                        "name": node.name,
                    }
                    for node in unhealthy_dynamic_nodes
                ],
            },
        )
        self.publish_event(
            "INFO",
            "Dynamic node failing scheduler health check",
            event_type="dynamic-node-health-check-failure",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=(
                {
                    "detail": {
                        "node": node.description(),
                    }
                }
                for node in unhealthy_dynamic_nodes
            ),
        )

    @log_exception(
        logger, "publish_unhealthy_dynamic_node_action_events", catch_exception=Exception, raise_on_error=False
    )
    def publish_unhealthy_dynamic_node_action_events(
        self, instances_to_terminate: List[str], power_down_nodes: List[str]
    ):
        self.publish_event(
            "INFO" if instances_to_terminate else "DEBUG",
            "Number of instances being terminating due to backing unhealthy dynamic nodes",
            event_type="dynamic-node-health-check-failure-instance-terminate-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "count": len(instances_to_terminate),
                "instances": [{"id": instance_id} for instance_id in instances_to_terminate],
            },
        )
        self.publish_event(
            "INFO" if power_down_nodes else "DEBUG",
            "Number of unhealthy dynamic nodes set to down and power_down",
            event_type="dynamic-node-health-check-failure-power-down-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={"count": len(power_down_nodes), "nodes": [{"name": node_name} for node_name in power_down_nodes]},
        )

    @log_exception(logger, "publish_unhealthy_static_node_events", catch_exception=Exception, raise_on_error=False)
    def publish_unhealthy_static_node_events(self, unhealthy_static_nodes: List[SlurmNode]):
        self.publish_event(
            "WARNING" if unhealthy_static_nodes else "DEGUG",
            "Number of static nodes failing scheduler health check",
            event_type="unhealthy-static-node-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "count": len(unhealthy_static_nodes),
                "nodes": [{"name": node.name} for node in unhealthy_static_nodes],
            },
        )
        self.publish_event(
            "INFO",
            "Static node failing scheduler health check",
            event_type="unhealthy-static-node",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=({"detail": {"node": node.description()}} for node in unhealthy_static_nodes),
        )

    @log_exception(logger, "publish_static_nodes_in_replacement", catch_exception=Exception, raise_on_error=False)
    def publish_static_nodes_in_replacement(
        self,
        unhealthy_static_nodes: List[SlurmNode],
        nodes_in_replacement: List[str],
        failed_nodes: Dict[str, List[str]],
    ):
        self.publish_event(
            "INFO" if nodes_in_replacement else "DEBUG",
            "After node maintenance, nodes currently in replacement",
            event_type="static-nodes-in-replacement-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "count": len(nodes_in_replacement),
                "nodes": [{"name": node_name} for node_name in nodes_in_replacement],
            },
        )
        self.publish_event(
            "DEBUG",
            "After node maintenance, node currently in replacement",
            event_type="static-node-in-replacement",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=(
                {
                    "detail": {
                        "node": node.description(),
                    }
                }
                for node in unhealthy_static_nodes
                if node.name in nodes_in_replacement
            ),
        )
        for error_code, failed_node_list in failed_nodes.items():
            self.publish_event(
                "WARNING",
                "Number of static nodes that failed replacement after node maintenance",
                event_type="static-nodes-in-replacement-failure-count",
                timestamp=ClusterEventPublisher.timestamp(),
                detail={
                    "error-code": error_code,
                    "count": len(failed_node_list),
                    "nodes": [{"name": node_name} for node_name in failed_node_list],
                },
            )
            self.publish_event(
                "DEBUG",
                "After node maintenance, node failed replacement",
                event_type="static-node-in-replacement-failure",
                timestamp=ClusterEventPublisher.timestamp(),
                event_supplier=(
                    {
                        "detail": {
                            "node": node.description(),
                            "error-code": error_code,
                        }
                    }
                    for node in unhealthy_static_nodes
                    if node.name in failed_node_list
                ),
            )

    @log_exception(logger, "publish_insufficient_capacity_events", catch_exception=Exception, raise_on_error=False)
    def publish_insufficient_capacity_events(
        self, ice_compute_resources_and_nodes_map: Dict[str, Dict[str, List[DynamicNode]]]
    ):
        self.publish_event(
            "WARNING" if ice_compute_resources_and_nodes_map else "DEBUG",
            "Insufficient capacity error count",
            event_type="insufficient-capacity-error-count",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=self._count_insufficient_capacity_errors(ice_compute_resources_and_nodes_map),
        )

        self.publish_event(
            "INFO" if ice_compute_resources_and_nodes_map else "DEBUG",
            "Insufficient capacity errors",
            event_type="insufficient-capacity-errors",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=self._flatten_insufficient_capacity_errors(ice_compute_resources_and_nodes_map),
        )

        self.publish_event(
            "DEBUG",
            "Node with insufficient capacity",
            event_type="insufficient-capacity-node",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=self._flatten_insufficient_capacity_nodes(ice_compute_resources_and_nodes_map),
        )

    @log_exception(logger, "publish_orphaned_instance_events", catch_exception=Exception, raise_on_error=False)
    def publish_orphaned_instance_events(self, cluster_instances: List[any], instances_to_terminate: List[str]):
        self.publish_event(
            "WARNING" if instances_to_terminate else "DEBUG",
            "Orphaned instance count",
            event_type="orphaned-instance-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "count": len(instances_to_terminate),
                "instances": [
                    {"id": instance.id} for instance in cluster_instances if instance.id in instances_to_terminate
                ],
            },
        )
        self.publish_event(
            "INFO",
            "Found orphaned instance",
            event_type="terminating-orphaned-instance",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=(
                {
                    "detail": {
                        "instance": instance.description(),
                    }
                }
                for instance in cluster_instances
                if instance.id in instances_to_terminate
            ),
        )

    @log_exception(logger, "publish_entering_protected_mode_events", catch_exception=Exception, raise_on_error=False)
    def publish_entering_protected_mode_events(self, partitions_protected_failure_count_map: Dict[str, Dict[str, int]]):
        self.publish_event(
            "WARNING",
            (
                "Setting cluster into protected mode due to failures detected in node provisioning. "
                "Please investigate the issue and then use "
                "'pcluster update-compute-fleet --status START_REQUESTED' command to re-enable the fleet."
            ),
            event_type="cluster-entering-protected-mode",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "partition_failures": partitions_protected_failure_count_map,
            },
        )
        self.publish_event(
            "INFO",
            "Partition compute resource failure count",
            event_type="partition-compute-resource-failure-count",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=self._flatten_partition_failure_counts(partitions_protected_failure_count_map),
        )

    @log_exception(logger, "publish_nodes_failing_health_check_events", catch_exception=Exception, raise_on_error=False)
    def publish_nodes_failing_health_check_events(
        self,
        health_check_type: str,
        nodes_failing_health_check: List[SlurmNode],
        rebooting_nodes: List[SlurmNode],
    ):
        self.publish_event(
            "WARNING" if nodes_failing_health_check else "DEBUG",
            f"Nodes failing {health_check_type} count",
            event_type="node-failed-health-check-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "health-check-type": health_check_type,
                "count": len(nodes_failing_health_check),
                "nodes": [{"name": node.name} for node in nodes_failing_health_check],
            },
        )
        self.publish_event(
            "INFO" if rebooting_nodes else "DEBUG",
            f"Rebooted nodes ignoring {health_check_type} count",
            event_type="rebooted-nodes-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "health-check-type": health_check_type,
                "count": len(rebooting_nodes),
                "nodes": [{"name": node.name} for node in rebooting_nodes],
            },
        )
        self.publish_event(
            "DEBUG",
            f"Node failing {health_check_type}, setting to DRAIN",
            event_type="node-failed-health-check",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=(
                {
                    "detail": {
                        "health-check-type": health_check_type,
                        "node": node.description(),
                    },
                }
                for node in nodes_failing_health_check
            ),
        )

    @log_exception(
        logger, "publish_failed_health_check_nodes_in_replacement", catch_exception=Exception, raise_on_error=False
    )
    def publish_failed_health_check_nodes_in_replacement(self, nodes_in_replacement: List[SlurmNode]):
        self.publish_event(
            "WARNING" if nodes_in_replacement else "DEBUG",
            "Number of static nodes in replacement that failed health checks",
            event_type="static-node-replacement-health-check-failure-count",
            timestamp=ClusterEventPublisher.timestamp(),
            detail={
                "count": len(nodes_in_replacement),
                "nodes": [
                    {
                        "name": node.name,
                    }
                    for node in nodes_in_replacement
                ],
            },
        )

        self.publish_event(
            "INFO",
            "Detected failed health check for static node in replacement - "
            "will attempt to replace node again immediately",
            event_type="static-node-replacement-health-check-failure",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=(
                {
                    "detail": {
                        "node": node.description(),
                    }
                }
                for node in nodes_in_replacement
            ),
        )

    @log_exception(logger, "publish_node_launch_events", catch_exception=Exception, raise_on_error=False)
    def publish_node_launch_events(self, successful_nodes: List[Tuple[str, any]], failed_nodes: Dict[str, List[str]]):
        def success_supplier():
            yield {
                "detail": {
                    "count": len(successful_nodes),
                    "nodes": [{"name": name, "id": instance.id} for name, instance in successful_nodes],
                },
            }

        self.publish_event(
            "INFO",
            "Number of successfully launched nodes",
            event_type="successful-launch-node-count",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=success_supplier(),
        )

        self.publish_event(
            "DEBUG",
            "Successfully Launched Node",
            event_type="successful-node-launch",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=(
                {
                    "detail": {
                        "node": {"name": node},
                        "instance": instance.description(),
                    }
                }
                for node, instance in successful_nodes
            ),
        )

        ice_failures: List(Tuple(str, List[str])) = []
        other_failures: List(Tuple(str, List[str])) = []

        for error_code, nodes in failed_nodes.items():
            (ice_failures if error_code in SlurmNode.EC2_ICE_ERROR_CODES else other_failures).append(
                (error_code, nodes)
            )

        ice_count = sum(len(nodes) for error_code, nodes in ice_failures)
        other_count = sum(len(nodes) for error_code, nodes in other_failures)
        total_failures = ice_count + other_count

        self.publish_event(
            "WARNING" if total_failures else "DEBUG",
            "Number of nodes that failed to launch",
            event_type="node-launch-failure-count",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=[
                {
                    "detail": {
                        "total": total_failures,
                        "ice-failures": {
                            "count": ice_count,
                            "errors": {
                                error_code: {"nodes": nodes, "count": len(nodes)} for error_code, nodes in ice_failures
                            },
                        }
                        if ice_count
                        else {"count": 0},
                        "other-failures": {
                            "count": other_count,
                            "errors": {
                                error_code: {"nodes": nodes, "count": len(nodes)}
                                for error_code, nodes in other_failures
                            },
                        }
                        if other_count
                        else {"count": 0},
                    },
                }
            ],
        )

        self.publish_event(
            "DEBUG",
            "Setting failed node to DOWN state",
            event_type="node-launch-failure",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=self._flatten_failed_launch_nodes(failed_nodes),
        )

    @log_exception(logger, "publish_handle_powering_down_nodes_events", catch_exception=Exception, raise_on_error=False)
    def publish_handle_powering_down_nodes_events(
        self, powering_down_nodes: List[SlurmNode], instances_to_terminate: List[str]
    ):
        def supply_nodes():
            yield {
                "detail": {
                    "count": len(instances_to_terminate),
                    "nodes": [
                        {"name": node.name, "id": node.instance.id}
                        for node in powering_down_nodes
                        if node.instance and node.instance.id in instances_to_terminate
                    ],
                }
            }

        self.publish_event(
            "INFO" if instances_to_terminate else "DEBUG",
            "Terminating instances that are backing powering down nodes",
            event_type="node-powering-down-instance-count",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=supply_nodes(),
        )

        self.publish_event(
            "DEBUG",
            "Terminating instance that is backing powering down node",
            event_type="node-powering-down-instance",
            timestamp=ClusterEventPublisher.timestamp(),
            event_supplier=(
                {
                    "detail": {
                        "node": node.description(),
                    }
                }
                for node in powering_down_nodes
                if node.instance and node.instance.id in instances_to_terminate
            ),
        )

    @staticmethod
    def _count_cluster_states(nodes: list[SlurmNode]):
        count_map = Counter((node.canonical_state_string for node in nodes))
        for state, count in count_map.items():
            yield {"detail": {"state": state, "count": count}}

    @staticmethod
    def _format_node_idle_time(max_node: SlurmNode, idle_nodes: List[SlurmNode]):
        return (
            {
                "idle-time": max_node.idle_time,
                "idle-count": len(idle_nodes),
                "most-idle-node": {
                    "name": max_node.name,
                    "partition": max_node.queue_name,
                    "resource": max_node.compute_resource_name,
                    "instance-id": max_node.instance.id if max_node.instance else None,
                },
            }
            if max_node and max_node.idle_time > 0
            else {"idle-time": 0, "idle-count": 0}
        )

    @staticmethod
    def _get_max_idle_times(nodes: list[SlurmNode]):
        idle_dynamic_nodes = []
        idle_static_nodes = []
        for node in nodes:
            if node.idle_time > 0:
                (idle_dynamic_nodes if isinstance(node, DynamicNode) else idle_static_nodes).append(node)

        dynamic_node_max = max((node for node in idle_dynamic_nodes), default=None, key=lambda node: node.idle_time)
        static_node_max = max((node for node in idle_static_nodes), default=None, key=lambda node: node.idle_time)

        yield {
            "detail": {
                "dynamic": ClusterEventPublisher._format_node_idle_time(dynamic_node_max, idle_dynamic_nodes),
                "static": ClusterEventPublisher._format_node_idle_time(static_node_max, idle_static_nodes),
            }
        }

    @staticmethod
    def _flatten_partition_failure_counts(partitions_failure_count_map: Dict[str, Dict[str, int]]):
        for partition, resources in partitions_failure_count_map.items():
            for resource, count in resources.items():
                yield {"detail": {"partition": partition, "resource": resource, "count": count}}

    @staticmethod
    def _count_insufficient_capacity_errors(ice_compute_resources_and_nodes_map: Dict[str, Dict[str, List[SlurmNode]]]):
        def extract_nodes():
            for resources in ice_compute_resources_and_nodes_map.values():
                for nodes in resources.values():
                    for node in nodes:
                        yield {
                            "name": node.name,
                        }

        yield {
            "detail": {
                "count": sum(
                    (
                        sum(len(nodes) for nodes in resource)
                        for resource in (
                            resources.values() for resources in ice_compute_resources_and_nodes_map.values()
                        )
                    )
                ),
                "nodes": list(extract_nodes()),
            }
        }

    @staticmethod
    def _flatten_insufficient_capacity_errors(
        ice_compute_resources_and_nodes_map: Dict[str, Dict[str, List[SlurmNode]]]
    ):
        for partition_name, resources in ice_compute_resources_and_nodes_map.items():
            for resource_name, nodes in resources.items():
                yield {
                    "detail": {
                        "partition": partition_name,
                        "resource": resource_name,
                        "error-code": nodes[0].error_code if nodes else None,
                        "count": len(nodes),
                        "nodes": [{"name": node.name} for node in nodes],
                    }
                }

    @staticmethod
    def _flatten_insufficient_capacity_nodes(
        ice_compute_resources_and_nodes_map: Dict[str, Dict[str, List[SlurmNode]]]
    ):
        for partition_name, resources in ice_compute_resources_and_nodes_map.items():
            for resource_name, nodes in resources.items():
                for node in nodes:
                    yield {
                        "detail": {
                            "partition": partition_name,
                            "resource": resource_name,
                            "error-code": node.error_code,
                            "node": node.description(),
                        }
                    }

    @staticmethod
    def _flatten_failed_launch_nodes(failed_nodes: Dict[str, List[str]]):
        for error_code, nodes in failed_nodes.items():
            for node_name in nodes:
                yield {
                    "detail": {
                        "error-code": error_code,
                        "node": {"name": node_name},
                    }
                }

import logging
import re
from collections import Counter
from datetime import datetime, timezone

# A nosec comment is appended to the following line in order to disable the B404 check.
# In this file the input of the module subprocess is trusted.
from typing import Callable, Dict, Iterable, List, Tuple

from slurm_plugin.common import log_exception
from slurm_plugin.slurm_resources import DynamicNode, SlurmNode

logger = logging.getLogger(__name__)

LAUNCH_FAILURE_GROUPING = {}
for failure in SlurmNode.EC2_ICE_ERROR_CODES:
    LAUNCH_FAILURE_GROUPING.update({failure: "ice-failures"})

for failure in ["VcpuLimitExceeded"]:
    LAUNCH_FAILURE_GROUPING.update({failure: "vcpu-limit-failures"})

for failure in ["VolumeLimitExceeded", "InsufficientVolumeCapacity"]:
    LAUNCH_FAILURE_GROUPING.update({failure: "volume-limit-failures"})

for failure in ["InvalidBlockDeviceMapping"]:
    LAUNCH_FAILURE_GROUPING.update({failure: "custom-ami-errors"})

for failure in ["UnauthorizedOperation", "AccessDeniedException"]:
    LAUNCH_FAILURE_GROUPING.update({failure: "iam-policy-errors"})


class ClusterEventPublisher:
    def __init__(self, event_publisher: Callable, max_list_size=100):
        self._publish_event = event_publisher
        self._max_list_size = max_list_size

    @property
    def publish_event(self):
        return self._publish_event

    @staticmethod
    def timestamp():
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    # ClusterMgtd Events
    @log_exception(logger, "publish_cluster_node_events", catch_exception=Exception, raise_on_error=False)
    def publish_cluster_node_events(self, cluster_nodes: List[SlurmNode]):
        timestamp = ClusterEventPublisher.timestamp()
        self.publish_event(
            "INFO",
            "Node State Counts",
            event_type="node-state-count",
            timestamp=timestamp,
            event_supplier=self._count_cluster_states(cluster_nodes),
        )
        self.publish_event(
            "INFO",
            "Idle node times",
            event_type="idle-node-time",
            timestamp=timestamp,
            event_supplier=self._get_max_idle_times(cluster_nodes),
        )
        self.publish_event(
            "DEBUG",
            "Node Info",
            event_type="node-info",
            timestamp=timestamp,
            event_supplier=({"detail": {"node": node.description()}} for node in cluster_nodes),
        )

    @log_exception(logger, "publish_powering_down_node_events", catch_exception=Exception, raise_on_error=False)
    def publish_powering_down_node_events(self, powering_down_nodes: List[SlurmNode]):
        timestamp = ClusterEventPublisher.timestamp()
        self.publish_event(
            "INFO" if powering_down_nodes else "DEBUG",
            "Powering down node count",
            event_type="node-powering-down-count",
            timestamp=timestamp,
            detail={"count": len(powering_down_nodes), "nodes": [{"name": node.name} for node in powering_down_nodes]},
        )
        self.publish_event(
            "DEBUG",
            "Powering down node",
            event_type="node-powering-down",
            timestamp=timestamp,
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
    def publish_unhealthy_dynamic_node_events(
        self,
        unhealthy_dynamic_nodes: Iterable[SlurmNode],
        instances_to_terminate: List[str],
        power_down_nodes: List[str],
    ):
        timestamp = ClusterEventPublisher.timestamp()

        self.publish_event(
            "WARNING" if unhealthy_dynamic_nodes else "DEBUG",
            "Number of dynamic nodes failing scheduler health check",
            event_type="dynamic-node-health-check-failure-count",
            timestamp=timestamp,
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
            "DEBUG",
            "Dynamic node failing scheduler health check",
            event_type="dynamic-node-health-check-failure",
            timestamp=timestamp,
            event_supplier=(
                {
                    "detail": {
                        "node": node.description(),
                    }
                }
                for node in unhealthy_dynamic_nodes
            ),
        )

        self.publish_event(
            "INFO" if instances_to_terminate else "DEBUG",
            "Number of instances being terminating due to backing unhealthy dynamic nodes",
            event_type="dynamic-node-failure-instance-terminate-count",
            timestamp=timestamp,
            detail={
                "count": len(instances_to_terminate),
                "instances": [{"id": instance_id} for instance_id in instances_to_terminate],
            },
        )
        self.publish_event(
            "INFO" if power_down_nodes else "DEBUG",
            "Number of unhealthy dynamic nodes set to down and power_down",
            event_type="dynamic-node-failure-power-down-count",
            timestamp=timestamp,
            detail={"count": len(power_down_nodes), "nodes": [{"name": node_name} for node_name in power_down_nodes]},
        )

    @log_exception(logger, "publish_unhealthy_static_node_events", catch_exception=Exception, raise_on_error=False)
    def publish_unhealthy_static_node_events(
        self,
        unhealthy_static_nodes: List[SlurmNode],
        instances_to_terminate: List[str],
        successful_nodes: List[Tuple[str, any]],
        nodes_in_replacement: List[str],
        failed_nodes: Dict[str, List[str]],
    ):
        timestamp = ClusterEventPublisher.timestamp()

        def terminated_instances_supplier():
            yield {
                "detail": {
                    "count": len(instances_to_terminate),
                    "nodes": [
                        {
                            "name": node.name,
                            "id": node.instance.id,
                            "ip": node.instance.private_ip,
                            "error-code": node.error_code,
                            "reason": node.reason,
                        }
                        for node in unhealthy_static_nodes
                        if node.instance
                    ],
                }
            }

        timestamp = ClusterEventPublisher.timestamp()
        self.publish_event(
            "INFO" if unhealthy_static_nodes else "DEGUG",
            "Number of static nodes failing scheduler health check",
            event_type="static-node-health-check-failure-count",
            timestamp=timestamp,
            detail={
                "count": len(unhealthy_static_nodes),
                "nodes": [{"name": node.name} for node in unhealthy_static_nodes],
            },
        )
        self.publish_event(
            "DEBUG",
            "Static node failing scheduler health check",
            event_type="static-node-health-check-failure",
            timestamp=timestamp,
            event_supplier=({"detail": {"node": node.description()}} for node in unhealthy_static_nodes),
        )

        self.publish_event(
            "INFO" if instances_to_terminate else "DEBUG",
            "Terminated instances backing unhealthy nodes",
            event_type="static-node-failure-instance-terminate-count",
            timestamp=timestamp,
            event_supplier=terminated_instances_supplier(),
        )

        self.publish_event(
            "INFO",
            "Number of successfully launched static nodes",
            event_type="successful-node-launch-count",
            timestamp=timestamp,
            event_supplier=ClusterEventPublisher._successful_node_launch_supplier(successful_nodes),
        )

        self.publish_event(
            "DEBUG",
            "Successfully launched static node",
            event_type="successful-node-launch",
            timestamp=timestamp,
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

        self.publish_event(
            "INFO" if nodes_in_replacement else "DEBUG",
            "After node maintenance, nodes currently in replacement",
            event_type="static-nodes-in-replacement-count",
            timestamp=timestamp,
            detail={
                "count": len(nodes_in_replacement),
                "nodes": [{"name": node_name} for node_name in nodes_in_replacement],
            },
        )

        self.publish_event(
            "DEBUG",
            "After node maintenance, node currently in replacement",
            event_type="static-node-in-replacement",
            timestamp=timestamp,
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

        self.publish_event(
            "WARNING" if failed_nodes else "DEBUG",
            "Number of static nodes that failed replacement after node maintenance",
            event_type="node-launch-failure-count",
            timestamp=timestamp,
            event_supplier=self._get_launch_failure_details(failed_nodes),
        )

        for error_code, failed_node_list in failed_nodes.items():
            self.publish_event(
                "DEBUG",
                "After node maintenance, node failed replacement",
                event_type="node-launch-failure",
                timestamp=timestamp,
                event_supplier=(
                    {
                        "detail": {
                            "node": node.description(),
                            "error-code": error_code,
                            "failure-type": ClusterEventPublisher._get_failure_type_from_error_code(error_code),
                        }
                    }
                    for node in unhealthy_static_nodes
                    if node.name in failed_node_list
                ),
            )

    @log_exception(logger, "publish_orphaned_instance_events", catch_exception=Exception, raise_on_error=False)
    def publish_orphaned_instance_events(self, cluster_instances: List[any], instances_to_terminate: List[str]):
        timestamp = ClusterEventPublisher.timestamp()
        self.publish_event(
            "WARNING" if instances_to_terminate else "DEBUG",
            "Orphaned instance count",
            event_type="orphaned-instance-count",
            timestamp=timestamp,
            detail={
                "count": len(instances_to_terminate),
                "instances": [
                    {"id": instance.id} for instance in cluster_instances if instance.id in instances_to_terminate
                ],
            },
        )
        self.publish_event(
            "DEBUG",
            "Orphaned instance",
            event_type="terminating-orphaned-instance",
            timestamp=timestamp,
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
        timestamp = ClusterEventPublisher.timestamp()
        self.publish_event(
            "WARNING",
            (
                "Setting cluster into protected mode due to failures detected in node provisioning. "
                "Please investigate the issue and then use "
                "'pcluster update-compute-fleet --status START_REQUESTED' command to re-enable the fleet."
            ),
            event_type="cluster-entering-protected-mode",
            timestamp=timestamp,
            detail={
                "partition_failures": partitions_protected_failure_count_map,
            },
        )
        self.publish_event(
            "DEBUG",
            "Partition compute resource failure count",
            event_type="partition-compute-resource-failure-count",
            timestamp=timestamp,
            event_supplier=self._flatten_partition_failure_counts(partitions_protected_failure_count_map),
        )

    @log_exception(logger, "publish_nodes_failing_health_check_events", catch_exception=Exception, raise_on_error=False)
    def publish_nodes_failing_health_check_events(
        self,
        health_check_type: str,
        nodes_failing_health_check: List[SlurmNode],
        rebooting_nodes: List[SlurmNode],
    ):
        timestamp = ClusterEventPublisher.timestamp()
        self.publish_event(
            "WARNING" if nodes_failing_health_check else "DEBUG",
            f"Seting nodes failing {health_check_type} to DRAIN",
            event_type="node-failed-health-check-count",
            timestamp=timestamp,
            detail={
                "health-check-type": health_check_type,
                "count": len(nodes_failing_health_check),
                "nodes": [{"name": node.name} for node in nodes_failing_health_check],
            },
        )
        self.publish_event(
            "INFO" if rebooting_nodes else "DEBUG",
            f"Ignoring recently rebooted nodes failing {health_check_type}",
            event_type="rebooted-nodes-count",
            timestamp=timestamp,
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
            timestamp=timestamp,
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
        timestamp = ClusterEventPublisher.timestamp()
        self.publish_event(
            "WARNING" if nodes_in_replacement else "DEBUG",
            "Number of static nodes in replacement that failed health checks - "
            "will attempt to replace these nodes again immediately",
            event_type="static-node-replacement-failure-count",
            timestamp=timestamp,
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
            "DEBUG",
            "Detected failed health check for static node in replacement - "
            "will attempt to replace node again immediately",
            event_type="static-node-replacement-health-check-failure",
            timestamp=timestamp,
            event_supplier=(
                {
                    "detail": {
                        "node": node.description(),
                    }
                }
                for node in nodes_in_replacement
            ),
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

        timestamp = ClusterEventPublisher.timestamp()
        self.publish_event(
            "INFO" if instances_to_terminate else "DEBUG",
            "Terminating instances that are backing powering down nodes",
            event_type="node-powering-down-instance-count",
            timestamp=timestamp,
            event_supplier=supply_nodes(),
        )

        self.publish_event(
            "DEBUG",
            "Terminating instance that is backing powering down node",
            event_type="node-powering-down-instance",
            timestamp=timestamp,
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

    # Slurm Resume Events
    @log_exception(
        logger, "publish_add_instance_for_nodes_success_events", catch_exception=Exception, raise_on_error=False
    )
    def publish_add_instance_for_nodes_success_events(self, successful_nodes: List[Tuple[str, any]]):
        timestamp = ClusterEventPublisher.timestamp()

        self.publish_event(
            "DEBUG",
            "Successfully launched node batch",
            event_type="successful-node-launch-batch-count",
            timestamp=timestamp,
            event_supplier=ClusterEventPublisher._successful_node_launch_supplier(successful_nodes),
        )

    @log_exception(
        logger, "publish_add_instance_for_nodes_failure_events", catch_exception=Exception, raise_on_error=False
    )
    def publish_add_instance_for_nodes_failure_events(
        self, error_code: str, error_message: str, failed_nodes: List[str]
    ):
        timestamp = ClusterEventPublisher.timestamp()

        def supply_details():
            yield {
                "detail": {
                    "count": len(failed_nodes) if failed_nodes else 0,
                    "error-code": error_code,
                    "error-message": error_message,
                    "nodes": [{"name": node} for node in failed_nodes],
                }
            }

        self.publish_event(
            "INFO",
            "Failed to launch a batch of nodes",
            event_type="batch-node-launch-failure-count",
            timestamp=timestamp,
            event_supplier=supply_details(),
        )

    @log_exception(logger, "publish_node_launch_events", catch_exception=Exception, raise_on_error=False)
    def publish_node_launch_events(self, successful_nodes: List[Tuple[str, any]], failed_nodes: Dict[str, List[str]]):
        timestamp = ClusterEventPublisher.timestamp()

        self.publish_event(
            "WARNING" if failed_nodes else "DEBUG",
            "Number of nodes that failed to launch",
            event_type="node-launch-failure-count",
            timestamp=timestamp,
            event_supplier=self._get_launch_failure_details(failed_nodes),
        )

        self.publish_event(
            "INFO",
            "Number of successfully launched nodes",
            event_type="successful-node-launch-count",
            timestamp=timestamp,
            event_supplier=ClusterEventPublisher._successful_node_launch_supplier(successful_nodes),
        )

        self.publish_event(
            "DEBUG",
            "Setting failed node to DOWN state",
            event_type="node-launch-failure",
            timestamp=timestamp,
            event_supplier=self._flatten_failed_launch_nodes(failed_nodes),
        )

        self.publish_event(
            "DEBUG",
            "Successfully Launched Node",
            event_type="successful-node-launch",
            timestamp=timestamp,
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

    # Slurm Suspend Events
    @log_exception(logger, "publish_node_launch_events", catch_exception=Exception, raise_on_error=False)
    def publish_suspend_events(self, slurm_node_spec: str):
        def detail_supplier():
            yield {"detail": {"nodes": list(_expand_slurm_node_spec(slurm_node_spec))}}

        self.publish_event("DEBUG", "Node Suspended", event_type="suspend-node", event_supplier=detail_supplier())

    @log_exception(logger, "publish_node_launch_events", catch_exception=Exception, raise_on_error=False)
    def publish_suspend_error_events(self, error_message, slurm_node_spec: str):
        def detail_supplier():
            yield {"detail": {"nodes": list(_expand_slurm_node_spec(slurm_node_spec))}}

        self.publish_event("ERROR", error_message, event_type="suspend-error", event_supplier=detail_supplier())

    @staticmethod
    def _count_cluster_states(nodes: list[SlurmNode]):
        count_map = Counter((node.canonical_state_string for node in nodes))
        for state, count in count_map.items():
            yield {"detail": {"state": state, "count": count}}

    @staticmethod
    def _successful_node_launch_supplier(successful_nodes: List[Tuple[str, any]]):
        yield {
            "detail": {
                "count": len(successful_nodes),
                "nodes": [
                    {"name": name, "id": instance.id, "ip": instance.private_ip} for name, instance in successful_nodes
                ],
            },
        }

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
                            "failure-type": ClusterEventPublisher._get_failure_type_from_error_code(node.error_code),
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
                        "failure-type": ClusterEventPublisher._get_failure_type_from_error_code(error_code),
                        "node": {"name": node_name},
                    }
                }

    @staticmethod
    def _get_launch_failure_details(failed_nodes: Dict[str, List[str]]) -> Dict:
        detail_map = {"other-failures": {"count": 0}}
        for failure_type in LAUNCH_FAILURE_GROUPING.values():
            detail_map.setdefault(failure_type, {"count": 0})

        total_failures = 0
        for error_code, nodes in failed_nodes.items():
            total_failures += len(nodes)
            failure_type = ClusterEventPublisher._get_failure_type_from_error_code(error_code)
            error_entry = detail_map.get(failure_type)
            error_entry.update({"count": error_entry.get("count") + len(nodes), error_code: list(nodes)})

        detail_map.update({"total": total_failures})

        yield {
            "detail": detail_map,
        }

    @staticmethod
    def _get_failure_type_from_error_code(error_code: str) -> str:
        return LAUNCH_FAILURE_GROUPING.get(error_code, "other-failures")


def _generate_from_range_spec(base_name: str, node_range_spec: str):
    """
    Expand a slurm range spec to individual node names.

    :param base_name: base name of node, e.g. 'queue1-dy-c5_xlarge-'
    :param node_range_spec: slurm node range spec, e.g. '1-3,7,11-12'
    :return: yields individual node names, e.g.
    ['queue1-dy-c5_xlarge-1', 'queue1-dy-c5_xlarge-2', 'queue1-dy-c5_xlarge-3']
    """
    span_list = re.split(r",", node_range_spec)
    for span in span_list:
        range_list = re.split(r"-", span)
        if len(range_list) < 0 or len(range_list) > 2:
            raise Exception(f"Invalid range spec: {node_range_spec}")
        start = int(range_list[0])
        end = int(range_list[1]) if len(range_list) > 1 else start
        if start > end:
            raise Exception(f"Invalid range spec: {node_range_spec}")
        for node_number in range(start, end + 1):
            yield f"{base_name}{node_number}"


def _expand_slurm_node_spec(slurm_node_spec: str):
    """
    Expand slurm nodelist notation into individual node names.

    Sample slurm nodelist notation:
    'queue1-dy-c5_xlarge-[1-3,7,11-12],queue2-st-t2_micro-5,queue3-dy-c5_large-[7,13-20]'.
    """
    slurm_node_spec = re.sub(r"\s+", "", slurm_node_spec)
    while slurm_node_spec:
        match = re.search(r"\[|,", slurm_node_spec)
        if match:
            end_pos = match.start()
            base_name = slurm_node_spec[:end_pos]
            if not base_name:
                raise Exception(f"Invalid node spec @{slurm_node_spec} - empty node name not allowed.")
            if slurm_node_spec[end_pos] == ",":
                slurm_node_spec = slurm_node_spec[match.end() :]
                yield base_name
            elif slurm_node_spec[end_pos] == "[":
                match = re.search(r"\],?", slurm_node_spec)
                if not match or match.start() < end_pos:
                    raise Exception(f"Invalid node spec @{slurm_node_spec} - missing closing ']'.")
                range_spec = slurm_node_spec[end_pos + 1 : match.start()]
                slurm_node_spec = slurm_node_spec[match.end() :]
                yield from _generate_from_range_spec(base_name, range_spec)
        else:
            yield slurm_node_spec
            break

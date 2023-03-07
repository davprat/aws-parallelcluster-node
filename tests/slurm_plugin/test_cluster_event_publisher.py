from datetime import datetime
from typing import Dict, List

import pytest
from assertpy import assert_that
from slurm_plugin.cluster_event_publisher import ClusterEventPublisher, _expand_slurm_node_spec
from slurm_plugin.fleet_manager import EC2Instance
from slurm_plugin.slurm_resources import DynamicNode, StaticNode


def event_handler(received_events: List[Dict], level_filter: List[str] = None):
    def _handler(level, message, event_type, *args, detail=None, **kwargs):
        if not level_filter or level in level_filter:
            if detail:
                received_events.append({event_type: detail})
            event_supplier = kwargs.get("event_supplier", [])
            for event in event_supplier:
                received_events.append({event_type: event.get("detail", None)})

    return _handler


@pytest.mark.parametrize(
    "test_nodes, expected_details, level_filter",
    [
        (
            [
                DynamicNode("queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"),
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),
                DynamicNode("queue1-dy-c4xlarge-1", "ip-1", "hostname", "DOWN", "queue1"),
                DynamicNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                    "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
            ],
            [
                {"node-state-count": {"state": "IDLE+CLOUD+POWERING_DOWN", "count": 1}},
                {"node-state-count": {"state": "IDLE+CLOUD", "count": 1}},
                {"node-state-count": {"state": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "count": 1}},
                {"node-state-count": {"state": "DOWN", "count": 1}},
                {"node-state-count": {"state": "COMPLETING+DRAIN", "count": 1}},
                {"node-state-count": {"state": "DOWN+CLOUD", "count": 2}},
                {
                    "idle-node-time": {
                        "dynamic": {"idle-time": 0, "idle-count": 0},
                        "static": {"idle-time": 0, "idle-count": 0},
                    }
                },
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_cluster_node_events(test_nodes, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter))

    # Run test
    event_publisher.publish_cluster_node_events(test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details, level_filter",
    [
        (
            [
                DynamicNode("queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"),
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),
                DynamicNode("queue1-dy-c4xlarge-1", "ip-1", "hostname", "DOWN", "queue1"),
                DynamicNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                    "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
            ],
            [
                {
                    "node-powering-down-count": {
                        "count": 7,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2"},
                            {"name": "queue-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c4xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-3"},
                            {"name": "queue2-dy-c5large-1"},
                            {"name": "queue2-dy-c5large-2"},
                        ],
                    }
                }
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_powering_down_node_events(test_nodes, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    # Run test
    event_publisher.publish_powering_down_node_events(test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details, level_filter",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2",
                    "ip-2",
                    "hostname",
                    "IDLE+CLOUD+POWERING_DOWN",
                    "queue1",
                    instance=EC2Instance("id-2", "ip-2", "hostname", "some_launch_time"),
                ),
                DynamicNode(
                    "queue-dy-c5xlarge-1",
                    "ip-3",
                    "hostname",
                    "IDLE+CLOUD",
                    "queue",
                    instance=EC2Instance("id-1", "ip-1", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                ),
                DynamicNode(
                    "queue1-dy-c5xlarge-1",
                    "ip-1",
                    "hostname",
                    "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                    "queue1",
                    instance=EC2Instance("id-2", "ip-4", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                ),
                DynamicNode(
                    "queue1-dy-c4xlarge-1",
                    "ip-1",
                    "hostname",
                    "DOWN",
                    "queue1",
                    instance=EC2Instance("id-5", "ip-5", "hostname", "some_launch_time"),
                ),
                DynamicNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                    "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                    instance=EC2Instance("id-6", "ip-6", "hostname", "some_launch_time"),
                ),
                DynamicNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
            ],
            [
                {
                    "node-powering-down-instance-count": {
                        "count": 5,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2", "id": "id-2"},
                            {"name": "queue-dy-c5xlarge-1", "id": "id-1"},
                            {"name": "queue1-dy-c5xlarge-1", "id": "id-2"},
                            {"name": "queue1-dy-c4xlarge-1", "id": "id-5"},
                            {"name": "queue1-dy-c5xlarge-3", "id": "id-6"},
                        ],
                    }
                }
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_handle_powering_down_node_events(test_nodes, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    instances_to_terminate = [node.instance.id for node in test_nodes if node.instance]

    # Run test
    event_publisher.publish_handle_powering_down_nodes_events(test_nodes, instances_to_terminate)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details, level_filter",
    [
        (
            [
                DynamicNode("queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"),
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),
                DynamicNode("queue1-dy-c4xlarge-1", "ip-1", "hostname", "DOWN", "queue1"),
                DynamicNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                    "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
            ],
            [
                {
                    "dynamic-node-health-check-failure-count": {
                        "count": 7,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2"},
                            {"name": "queue-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c4xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-3"},
                            {"name": "queue2-dy-c5large-1"},
                            {"name": "queue2-dy-c5large-2"},
                        ],
                    }
                },
                {
                    "dynamic-node-failure-instance-terminate-count": {
                        "count": 7,
                        "instances": [
                            {"id": "instance-queue1-dy-c5xlarge-2"},
                            {"id": "instance-queue-dy-c5xlarge-1"},
                            {"id": "instance-queue1-dy-c5xlarge-1"},
                            {"id": "instance-queue1-dy-c4xlarge-1"},
                            {"id": "instance-queue1-dy-c5xlarge-3"},
                            {"id": "instance-queue2-dy-c5large-1"},
                            {"id": "instance-queue2-dy-c5large-2"},
                        ],
                    }
                },
                {
                    "dynamic-node-failure-power-down-count": {
                        "count": 7,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2"},
                            {"name": "queue-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c4xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-3"},
                            {"name": "queue2-dy-c5large-1"},
                            {"name": "queue2-dy-c5large-2"},
                        ],
                    }
                },
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_unhealthy_dynamic_node_events(test_nodes, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    instances_to_terminate = [f"instance-{node.name}" for node in test_nodes]
    power_down_nodes = [node.name for node in test_nodes]

    # Run test
    event_publisher.publish_unhealthy_dynamic_node_events(test_nodes, instances_to_terminate, power_down_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details, level_filter",
    [
        (
            [
                StaticNode("queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"),
                StaticNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                StaticNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),
                StaticNode("queue1-dy-c4xlarge-1", "ip-1", "hostname", "DOWN", "queue1"),
                StaticNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                    "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes",
                ),
                StaticNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Failure when resuming nodes",
                ),
                StaticNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Error",
                ),
                StaticNode(
                    "queue2-dy-c5large-3",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:UnauthorizedOperation)Error",
                ),
                StaticNode(
                    "queue2-dy-c5large-4",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InvalidBlockDeviceMapping)Error",
                ),
                StaticNode(
                    "queue2-dy-c5large-5",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:AccessDeniedException)Error",
                ),
                StaticNode(
                    "queue2-dy-c5large-6",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:VcpuLimitExceeded)Error",
                ),
                StaticNode(
                    "queue2-dy-c5large-8",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:VolumeLimitExceeded)Error",
                ),
                StaticNode(
                    "queue2-dy-c5large-8",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientVolumeCapacity)Error",
                ),
            ],
            [
                {
                    "static-node-health-check-failure-count": {
                        "count": 13,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2"},
                            {"name": "queue-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c4xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-3"},
                            {"name": "queue2-dy-c5large-1"},
                            {"name": "queue2-dy-c5large-2"},
                            {"name": "queue2-dy-c5large-3"},
                            {"name": "queue2-dy-c5large-4"},
                            {"name": "queue2-dy-c5large-5"},
                            {"name": "queue2-dy-c5large-6"},
                            {"name": "queue2-dy-c5large-8"},
                            {"name": "queue2-dy-c5large-8"},
                        ],
                    }
                },
                {
                    "static-node-failure-instance-terminate-count": {
                        "count": 13,
                        "nodes": [
                            {
                                "name": "queue1-dy-c5xlarge-2",
                                "id": "i-id-0",
                                "ip": "1.2.3.0",
                                "error-code": None,
                                "reason": None,
                            },
                            {
                                "name": "queue-dy-c5xlarge-1",
                                "id": "i-id-1",
                                "ip": "1.2.3.1",
                                "error-code": None,
                                "reason": None,
                            },
                            {
                                "name": "queue1-dy-c5xlarge-1",
                                "id": "i-id-2",
                                "ip": "1.2.3.2",
                                "error-code": None,
                                "reason": None,
                            },
                            {
                                "name": "queue1-dy-c4xlarge-1",
                                "id": "i-id-3",
                                "ip": "1.2.3.3",
                                "error-code": None,
                                "reason": None,
                            },
                            {
                                "name": "queue1-dy-c5xlarge-3",
                                "id": "i-id-4",
                                "ip": "1.2.3.4",
                                "error-code": "InsufficientReservedInstanceCapacity",
                                "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes",
                            },
                            {
                                "name": "queue2-dy-c5large-1",
                                "id": "i-id-5",
                                "ip": "1.2.3.5",
                                "error-code": "InsufficientHostCapacity",
                                "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes",
                            },
                            {
                                "name": "queue2-dy-c5large-2",
                                "id": "i-id-6",
                                "ip": "1.2.3.6",
                                "error-code": "InsufficientHostCapacity",
                                "reason": "(Code:InsufficientHostCapacity)Error",
                            },
                            {
                                "name": "queue2-dy-c5large-3",
                                "id": "i-id-7",
                                "ip": "1.2.3.7",
                                "error-code": "UnauthorizedOperation",
                                "reason": "(Code:UnauthorizedOperation)Error",
                            },
                            {
                                "name": "queue2-dy-c5large-4",
                                "id": "i-id-8",
                                "ip": "1.2.3.8",
                                "error-code": "InvalidBlockDeviceMapping",
                                "reason": "(Code:InvalidBlockDeviceMapping)Error",
                            },
                            {
                                "name": "queue2-dy-c5large-5",
                                "id": "i-id-9",
                                "ip": "1.2.3.9",
                                "error-code": "AccessDeniedException",
                                "reason": "(Code:AccessDeniedException)Error",
                            },
                            {
                                "name": "queue2-dy-c5large-6",
                                "id": "i-id-10",
                                "ip": "1.2.3.10",
                                "error-code": "VcpuLimitExceeded",
                                "reason": "(Code:VcpuLimitExceeded)Error",
                            },
                            {
                                "name": "queue2-dy-c5large-8",
                                "id": "i-id-11",
                                "ip": "1.2.3.11",
                                "error-code": "VolumeLimitExceeded",
                                "reason": "(Code:VolumeLimitExceeded)Error",
                            },
                            {
                                "name": "queue2-dy-c5large-8",
                                "id": "i-id-12",
                                "ip": "1.2.3.12",
                                "error-code": "InsufficientVolumeCapacity",
                                "reason": "(Code:InsufficientVolumeCapacity)Error",
                            },
                        ],
                    }
                },
                {
                    "successful-node-launch-count": {
                        "count": 4,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2", "id": "i-id-0", "ip": "1.2.3.0"},
                            {"name": "queue-dy-c5xlarge-1", "id": "i-id-1", "ip": "1.2.3.1"},
                            {"name": "queue1-dy-c5xlarge-1", "id": "i-id-2", "ip": "1.2.3.2"},
                            {"name": "queue1-dy-c4xlarge-1", "id": "i-id-3", "ip": "1.2.3.3"},
                        ],
                    }
                },
                {
                    "static-nodes-in-replacement-count": {
                        "count": 13,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2"},
                            {"name": "queue-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c4xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-3"},
                            {"name": "queue2-dy-c5large-1"},
                            {"name": "queue2-dy-c5large-2"},
                            {"name": "queue2-dy-c5large-3"},
                            {"name": "queue2-dy-c5large-4"},
                            {"name": "queue2-dy-c5large-5"},
                            {"name": "queue2-dy-c5large-6"},
                            {"name": "queue2-dy-c5large-8"},
                            {"name": "queue2-dy-c5large-8"},
                        ],
                    }
                },
                {
                    "node-launch-failure-count": {
                        "other-failures": {"count": 0},
                        "ice-failures": {
                            "count": 3,
                            "InsufficientReservedInstanceCapacity": ["queue1-dy-c5xlarge-3"],
                            "InsufficientHostCapacity": ["queue2-dy-c5large-1", "queue2-dy-c5large-2"],
                        },
                        "vcpu-limit-failures": {"count": 1, "VcpuLimitExceeded": ["queue2-dy-c5large-6"]},
                        "volume-limit-failures": {
                            "count": 2,
                            "VolumeLimitExceeded": ["queue2-dy-c5large-8"],
                            "InsufficientVolumeCapacity": ["queue2-dy-c5large-8"],
                        },
                        "custom-ami-errors": {"count": 1, "InvalidBlockDeviceMapping": ["queue2-dy-c5large-4"]},
                        "iam-policy-errors": {
                            "count": 2,
                            "UnauthorizedOperation": ["queue2-dy-c5large-3"],
                            "AccessDeniedException": ["queue2-dy-c5large-5"],
                        },
                        "total": 9,
                    }
                },
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_unhealthy_static_node_events(test_nodes, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    instances = [
        EC2Instance(f"i-id-{instance_id}", f"1.2.3.{instance_id}", f"host-{instance_id}", "sometime")
        for instance_id in range(len(test_nodes))
    ]

    nodes_and_instances = zip(test_nodes, instances)

    for node, instance in nodes_and_instances:
        node.instance = instance

    nodes_in_replacement = [node.name for node in test_nodes]
    failed_nodes = {}
    for node in test_nodes:
        if node.error_code:
            failed_nodes.setdefault(node.error_code, []).append(node.name)

    success_nodes = [(node.name, node.instance) for node in test_nodes if not node.error_code and node.instance]

    # Run test
    event_publisher.publish_unhealthy_static_node_events(
        test_nodes,
        [node.instance.id for node in test_nodes if node.instance],
        success_nodes,
        nodes_in_replacement,
        failed_nodes,
    )

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_instances, expected_details, level_filter",
    [
        (
            [
                EC2Instance("id-2", "ip-2", "hostname", "some_launch_time"),
                EC2Instance("id-1", "ip-1", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-2", "ip-4", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-5", "ip-5", "hostname", "some_launch_time"),
                EC2Instance("id-6", "ip-6", "hostname", "some_launch_time"),
            ],
            [
                {
                    "orphaned-instance-count": {
                        "count": 5,
                        "instances": [{"id": "id-2"}, {"id": "id-1"}, {"id": "id-2"}, {"id": "id-5"}, {"id": "id-6"}],
                    }
                },
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_orphaned_instance_events(test_instances, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    instances_to_terminate = [instance.id for instance in test_instances]

    # Run test
    event_publisher.publish_orphaned_instance_events(test_instances, instances_to_terminate)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_map, expected_details, level_filter",
    [
        (
            {
                "partition-1": {
                    "resource-1a": 3,
                    "resource-1b": 2,
                },
                "partition-2": {
                    "resource-2a": 1,
                    "resource-2b": 7,
                    "resource-2c": 3,
                },
            },
            [
                {
                    "cluster-entering-protected-mode": {
                        "partition_failures": {
                            "partition-1": {"resource-1a": 3, "resource-1b": 2},
                            "partition-2": {"resource-2a": 1, "resource-2b": 7, "resource-2c": 3},
                        }
                    }
                },
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_entering_protected_mode_events(test_map, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    # Run test
    event_publisher.publish_entering_protected_mode_events(test_map)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details, level_filter",
    [
        (
            [
                DynamicNode("queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"),
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),
                DynamicNode("queue1-dy-c4xlarge-1", "ip-1", "hostname", "DOWN", "queue1"),
                DynamicNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                    "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
            ],
            [
                {
                    "node-failed-health-check-count": {
                        "health-check-type": "TriedToWalkInsteadOfRun",
                        "count": 7,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2"},
                            {"name": "queue-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c4xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-3"},
                            {"name": "queue2-dy-c5large-1"},
                            {"name": "queue2-dy-c5large-2"},
                        ],
                    }
                },
                {
                    "rebooted-nodes-count": {
                        "health-check-type": "TriedToWalkInsteadOfRun",
                        "count": 7,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2"},
                            {"name": "queue-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c4xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-3"},
                            {"name": "queue2-dy-c5large-1"},
                            {"name": "queue2-dy-c5large-2"},
                        ],
                    }
                },
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_nodes_failing_health_check_events(test_nodes, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    # Run test
    event_publisher.publish_nodes_failing_health_check_events("TriedToWalkInsteadOfRun", test_nodes, test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details, level_filter",
    [
        (
            [
                DynamicNode("queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"),
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),
                DynamicNode("queue1-dy-c4xlarge-1", "ip-1", "hostname", "DOWN", "queue1"),
                DynamicNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                    "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                DynamicNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
            ],
            [
                {
                    "static-node-replacement-failure-count": {
                        "count": 7,
                        "nodes": [
                            {"name": "queue1-dy-c5xlarge-2"},
                            {"name": "queue-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-1"},
                            {"name": "queue1-dy-c4xlarge-1"},
                            {"name": "queue1-dy-c5xlarge-3"},
                            {"name": "queue2-dy-c5large-1"},
                            {"name": "queue2-dy-c5large-2"},
                        ],
                    }
                }
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_failed_health_check_nodes_in_replacement(test_nodes, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    # Run test
    event_publisher.publish_failed_health_check_nodes_in_replacement(test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_instances, failed_nodes, expected_details, level_filter",
    [
        (
            [
                EC2Instance("id-2", "ip-2", "hostname", "some_launch_time"),
                EC2Instance("id-1", "ip-1", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-2", "ip-4", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-5", "ip-5", "hostname", "some_launch_time"),
                EC2Instance("id-6", "ip-6", "hostname", "some_launch_time"),
            ],
            {
                "Error1": [
                    "node-a-1",
                    "node-a-2",
                    "node-a-3",
                ],
                "Error2": [
                    "node-b-1",
                    "node-b-2",
                ],
                "InsufficientInstanceCapacity": [
                    "ice-a-1",
                    "ice-a-2",
                    "ice-a-3",
                ],
                "InsufficientHostCapacity": [
                    "ice-b-1",
                    "ice-b-2",
                ],
                "InsufficientReservedInstanceCapacity": [
                    "ice-c-1",
                    "ice-c-2",
                    "ice-c-3",
                ],
                "MaxSpotInstanceCountExceeded": [
                    "ice-d-1",
                    "ice-d-2",
                ],
                "Unsupported": [
                    "ice-e-1",
                    "ice-e-2",
                    "ice-e-3",
                ],
                "SpotMaxPriceTooLow": [
                    "ice-f-1",
                    "ice-f-2",
                ],
                "VcpuLimitExceeded": [
                    "vcpu-g-1",
                ],
                "VolumeLimitExceeded": [
                    "vle-h-1",
                    "vle-h-2",
                ],
                "InsufficientVolumeCapacity": [
                    "ivc-i-1",
                    "ivc-i-2",
                    "ivc-i-3",
                ],
                "InvalidBlockDeviceMapping": [
                    "ibdm-j-1",
                    "ibdm-j-2",
                    "ibdm-j-3",
                ],
                "UnauthorizedOperation": [
                    "iam-k-1",
                    "iam-k-2",
                ],
                "AccessDeniedException": [
                    "iam-l-1",
                ],
            },
            [
                {
                    "node-launch-failure-count": {
                        "other-failures": {
                            "count": 5,
                            "Error1": ["node-a-1", "node-a-2", "node-a-3"],
                            "Error2": ["node-b-1", "node-b-2"],
                        },
                        "ice-failures": {
                            "count": 15,
                            "InsufficientInstanceCapacity": ["ice-a-1", "ice-a-2", "ice-a-3"],
                            "InsufficientHostCapacity": ["ice-b-1", "ice-b-2"],
                            "InsufficientReservedInstanceCapacity": ["ice-c-1", "ice-c-2", "ice-c-3"],
                            "MaxSpotInstanceCountExceeded": ["ice-d-1", "ice-d-2"],
                            "Unsupported": ["ice-e-1", "ice-e-2", "ice-e-3"],
                            "SpotMaxPriceTooLow": ["ice-f-1", "ice-f-2"],
                        },
                        "vcpu-limit-failures": {"count": 1, "VcpuLimitExceeded": ["vcpu-g-1"]},
                        "volume-limit-failures": {
                            "count": 5,
                            "VolumeLimitExceeded": ["vle-h-1", "vle-h-2"],
                            "InsufficientVolumeCapacity": ["ivc-i-1", "ivc-i-2", "ivc-i-3"],
                        },
                        "custom-ami-errors": {
                            "count": 3,
                            "InvalidBlockDeviceMapping": ["ibdm-j-1", "ibdm-j-2", "ibdm-j-3"],
                        },
                        "iam-policy-errors": {
                            "count": 3,
                            "UnauthorizedOperation": ["iam-k-1", "iam-k-2"],
                            "AccessDeniedException": ["iam-l-1"],
                        },
                        "total": 32,
                    }
                },
                {
                    "successful-node-launch-count": {
                        "count": 5,
                        "nodes": [
                            {"name": "node-id-2", "id": "id-2", "ip": "ip-2"},
                            {"name": "node-id-1", "id": "id-1", "ip": "ip-1"},
                            {"name": "node-id-2", "id": "id-2", "ip": "ip-4"},
                            {"name": "node-id-5", "id": "id-5", "ip": "ip-5"},
                            {"name": "node-id-6", "id": "id-6", "ip": "ip-6"},
                        ],
                    }
                },
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
        (
            [
                EC2Instance("id-2", "ip-2", "hostname", "some_launch_time"),
                EC2Instance("id-1", "ip-1", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-2", "ip-4", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-5", "ip-5", "hostname", "some_launch_time"),
                EC2Instance("id-6", "ip-6", "hostname", "some_launch_time"),
            ],
            {},
            [
                {
                    "successful-node-launch-count": {
                        "count": 5,
                        "nodes": [
                            {"name": "node-id-2", "id": "id-2", "ip": "ip-2"},
                            {"name": "node-id-1", "id": "id-1", "ip": "ip-1"},
                            {"name": "node-id-2", "id": "id-2", "ip": "ip-4"},
                            {"name": "node-id-5", "id": "id-5", "ip": "ip-5"},
                            {"name": "node-id-6", "id": "id-6", "ip": "ip-6"},
                        ],
                    }
                }
            ],
            ["ERROR", "WARNING", "INFO"],
        ),
    ],
)
def test_publish_node_launch_events(test_instances, failed_nodes, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    successful_nodes = [(f"node-{instance.id}", instance) for instance in test_instances]

    # Run test
    event_publisher.publish_node_launch_events(successful_nodes, failed_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_instances, expected_details, level_filter",
    [
        (
            [
                EC2Instance("id-2", "ip-2", "hostname", "some_launch_time"),
                EC2Instance("id-1", "ip-1", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-2", "ip-4", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-5", "ip-5", "hostname", "some_launch_time"),
                EC2Instance("id-6", "ip-6", "hostname", "some_launch_time"),
            ],
            [
                {
                    "successful-node-launch-batch-count": {
                        "count": 5,
                        "nodes": [
                            {"name": "node-id-2", "id": "id-2", "ip": "ip-2"},
                            {"name": "node-id-1", "id": "id-1", "ip": "ip-1"},
                            {"name": "node-id-2", "id": "id-2", "ip": "ip-4"},
                            {"name": "node-id-5", "id": "id-5", "ip": "ip-5"},
                            {"name": "node-id-6", "id": "id-6", "ip": "ip-6"},
                        ],
                    }
                }
            ],
            [],
        )
    ],
)
def test_publish_add_instance_for_nodes_success_events(test_instances, expected_details, level_filter):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    successful_nodes = [(f"node-{instance.id}", instance) for instance in test_instances]

    # Run test
    event_publisher.publish_add_instance_for_nodes_success_events(successful_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "error_code, error_message, failed_nodes, expected_details, level_filter",
    [
        (
            "ItDidNotWork",
            "Your call failed",
            [
                "ab-dy-1",
                "ab-dy-2",
                "ab-dy-3",
                "ab-dy-4",
            ],
            [
                {
                    "batch-node-launch-failure-count": {
                        "count": 4,
                        "error-code": "ItDidNotWork",
                        "error-message": "Your call failed",
                        "nodes": [{"name": "ab-dy-1"}, {"name": "ab-dy-2"}, {"name": "ab-dy-3"}, {"name": "ab-dy-4"}],
                    }
                }
            ],
            [],
        )
    ],
)
def test_publish_add_instance_for_nodes_failure_events(
    error_code, error_message, failed_nodes, expected_details, level_filter
):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events, level_filter=level_filter))

    # Run test
    event_publisher.publish_add_instance_for_nodes_failure_events(error_code, error_message, failed_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "node_spec, expected_node_list",
    [
        (
            "",
            [],
        ),
        (
            "queue1-dy-c5_xlarge-1",
            [
                "queue1-dy-c5_xlarge-1",
            ],
        ),
        (
            "queue1-dy-c5_xlarge-[1-3]",
            [
                "queue1-dy-c5_xlarge-1",
                "queue1-dy-c5_xlarge-2",
                "queue1-dy-c5_xlarge-3",
            ],
        ),
        (
            "queue1-dy-c5_xlarge-1,queue1-dy-c5_xlarge-2,queue1-dy-c5_xlarge-3",
            [
                "queue1-dy-c5_xlarge-1",
                "queue1-dy-c5_xlarge-2",
                "queue1-dy-c5_xlarge-3",
            ],
        ),
        (
            "queue1-dy-c5_xlarge-[1-2,4,6,8-9]",
            [
                "queue1-dy-c5_xlarge-1",
                "queue1-dy-c5_xlarge-2",
                "queue1-dy-c5_xlarge-4",
                "queue1-dy-c5_xlarge-6",
                "queue1-dy-c5_xlarge-8",
                "queue1-dy-c5_xlarge-9",
            ],
        ),
        (
            "queue1-dy-c5_xlarge-[1-2,4,6,8-9],queue1-dy-c6_xlarge-6",
            [
                "queue1-dy-c5_xlarge-1",
                "queue1-dy-c5_xlarge-2",
                "queue1-dy-c5_xlarge-4",
                "queue1-dy-c5_xlarge-6",
                "queue1-dy-c5_xlarge-8",
                "queue1-dy-c5_xlarge-9",
                "queue1-dy-c6_xlarge-6",
            ],
        ),
        (
            "queue1-dy-c5_xlarge-[1-2,4,6,8-9],queue1-dy-c6_xlarge-[1,4-6,8]",
            [
                "queue1-dy-c5_xlarge-1",
                "queue1-dy-c5_xlarge-2",
                "queue1-dy-c5_xlarge-4",
                "queue1-dy-c5_xlarge-6",
                "queue1-dy-c5_xlarge-8",
                "queue1-dy-c5_xlarge-9",
                "queue1-dy-c6_xlarge-1",
                "queue1-dy-c6_xlarge-4",
                "queue1-dy-c6_xlarge-5",
                "queue1-dy-c6_xlarge-6",
                "queue1-dy-c6_xlarge-8",
            ],
        ),
        (
            "queue1-dy-c6_xlarge-2,queue1-dy-c5_xlarge-[1-2,4,6,8-9],queue1-dy-c6_xlarge-6",
            [
                "queue1-dy-c6_xlarge-2",
                "queue1-dy-c5_xlarge-1",
                "queue1-dy-c5_xlarge-2",
                "queue1-dy-c5_xlarge-4",
                "queue1-dy-c5_xlarge-6",
                "queue1-dy-c5_xlarge-8",
                "queue1-dy-c5_xlarge-9",
                "queue1-dy-c6_xlarge-6",
            ],
        ),
    ],
    ids=[
        "empty",
        "simple_name",
        "simple_range",
        "multiple_simple_names",
        "range_with_gaps",
        "range_and_simple_names",
        "mutliple_ranges",
        "mixed_simple_and_ranged",
    ],
)
def test_expand_slurm_node_spec(node_spec, expected_node_list):
    actual_node_list = list(_expand_slurm_node_spec(node_spec))
    assert_that(actual_node_list).is_length(len(expected_node_list))
    for node_name in expected_node_list:
        assert_that(actual_node_list).contains(node_name)


@pytest.mark.parametrize(
    "bad_input",
    [
        ",queue1-dy-c5_xlarge-[1-2,4,6,8-9],queue1-dy-c6_xlarge-6",
        "queue1-dy-c5_xlarge-[1-,4,6,8-9],queue1-dy-c6_xlarge-6",
        "queue1-dy-c5_xlarge-[-2,4,6,8-9],queue1-dy-c6_xlarge-6",
        "queue1-dy-c5_xlarge-[-2,4,6,8-9,queue1-dy-c6_xlarge-6",
        "queue1-dy-c5_xlarge-[-2,4,6,8-9],queue1-dy-c6_xlarge-6]",
        "queue1-dy-c5_xlarge-[1-2,4,6,8-9,queue1-dy-c6_xlarge-6]",
        "queue1-dy-c5_xlarge-[1-2,4,6,8-9,queue1]",
        "queue1-dy-c5_xlarge-[1-3-8]",
        "queue1-dy-c5_xlarge-[2-1]",
        "queue1-dy-c5_xlarge-1,,queue1-dy-c5_xlarge-2",
    ],
)
def test_expand_slurm_node_spec_raises_on_bad_input(bad_input):
    result = _expand_slurm_node_spec(bad_input)
    assert_that(list).raises(Exception).when_called_with(result)

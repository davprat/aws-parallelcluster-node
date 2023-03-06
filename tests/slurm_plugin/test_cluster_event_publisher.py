from datetime import datetime
from typing import Dict, List

import pytest
from assertpy import assert_that
from slurm_plugin.cluster_event_publisher import ClusterEventPublisher
from slurm_plugin.fleet_manager import EC2Instance
from slurm_plugin.slurm_resources import DynamicNode, StaticNode


def event_handler(received_events: List[Dict]):
    def _handler(*args, detail=None, **kwargs):
        if detail:
            received_events.append(detail)
        event_supplier = kwargs.get("event_supplier", [])
        for event in event_supplier:
            received_events.append(event.get("detail", None))

    return _handler


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
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
                {"state": "IDLE+CLOUD+POWERING_DOWN", "count": 1},
                {"state": "IDLE+CLOUD", "count": 1},
                {"state": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "count": 1},
                {"state": "DOWN", "count": 1},
                {"state": "COMPLETING+DRAIN", "count": 1},
                {"state": "DOWN+CLOUD", "count": 2},
                {"dynamic": {"idle-time": 0, "idle-count": 0}, "static": {"idle-time": 0, "idle-count": 0}},
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-2",
                        "address": "ip-2",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD+POWERING_DOWN",
                        "state": "IDLE",
                        "state-flags": ["CLOUD", "POWERING_DOWN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue-dy-c5xlarge-1",
                        "address": "ip-3",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD",
                        "state": "IDLE",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue"],
                        "queue-name": "queue",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                        "state": "MIXED",
                        "state-flags": ["CLOUD", "NOT_RESPONDING", "POWERING_UP"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c4xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "DOWN",
                        "state": "DOWN",
                        "state-flags": [],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c4xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    }
                },
            ],
        ),
    ],
)
def test_publish_cluster_node_events(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    # Run test
    event_publisher.publish_cluster_node_events(test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
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
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-2",
                        "address": "ip-2",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD+POWERING_DOWN",
                        "state": "IDLE",
                        "state-flags": ["CLOUD", "POWERING_DOWN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue-dy-c5xlarge-1",
                        "address": "ip-3",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD",
                        "state": "IDLE",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue"],
                        "queue-name": "queue",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                        "state": "MIXED",
                        "state-flags": ["CLOUD", "NOT_RESPONDING", "POWERING_UP"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c4xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "DOWN",
                        "state": "DOWN",
                        "state-flags": [],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c4xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    }
                },
            ],
        ),
    ],
)
def test_publish_powering_down_node_events(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    # Run test
    event_publisher.publish_powering_down_node_events(test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
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
                ),  # powering_down
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
                ),  # bootstrap failure dynamic
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
                    "count": 5,
                    "nodes": [
                        {"name": "queue1-dy-c5xlarge-2", "id": "id-2"},
                        {"name": "queue-dy-c5xlarge-1", "id": "id-1"},
                        {"name": "queue1-dy-c5xlarge-1", "id": "id-2"},
                        {"name": "queue1-dy-c4xlarge-1", "id": "id-5"},
                        {"name": "queue1-dy-c5xlarge-3", "id": "id-6"},
                    ],
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-2",
                        "address": "ip-2",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD+POWERING_DOWN",
                        "state": "IDLE",
                        "state-flags": ["CLOUD", "POWERING_DOWN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": {
                            "id": "id-2",
                            "private-ip": "ip-2",
                            "hostname": "hostname",
                            "launch-time": "some_launch_time",
                        },
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue-dy-c5xlarge-1",
                        "address": "ip-3",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD",
                        "state": "IDLE",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue"],
                        "queue-name": "queue",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": {
                            "id": "id-1",
                            "private-ip": "ip-1",
                            "hostname": "hostname",
                            "launch-time": "2020-01-01 00:00:00",
                        },
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                        "state": "MIXED",
                        "state-flags": ["CLOUD", "NOT_RESPONDING", "POWERING_UP"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": {
                            "id": "id-2",
                            "private-ip": "ip-4",
                            "hostname": "hostname",
                            "launch-time": "2020-01-01 00:00:00",
                        },
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c4xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "DOWN",
                        "state": "DOWN",
                        "state-flags": [],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c4xlarge",
                        "node-type": "dy",
                        "instance": {
                            "id": "id-5",
                            "private-ip": "ip-5",
                            "hostname": "hostname",
                            "launch-time": "some_launch_time",
                        },
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": {
                            "id": "id-6",
                            "private-ip": "ip-6",
                            "hostname": "hostname",
                            "launch-time": "some_launch_time",
                        },
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
            ],
        ),
    ],
)
def test_publish_handle_powering_down_node_events(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    instances_to_terminate = [node.instance.id for node in test_nodes if node.instance]

    # Run test
    event_publisher.publish_handle_powering_down_nodes_events(test_nodes, instances_to_terminate)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
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
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-2",
                        "address": "ip-2",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD+POWERING_DOWN",
                        "state": "IDLE",
                        "state-flags": ["CLOUD", "POWERING_DOWN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue-dy-c5xlarge-1",
                        "address": "ip-3",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD",
                        "state": "IDLE",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue"],
                        "queue-name": "queue",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                        "state": "MIXED",
                        "state-flags": ["CLOUD", "NOT_RESPONDING", "POWERING_UP"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c4xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "DOWN",
                        "state": "DOWN",
                        "state-flags": [],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c4xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    }
                },
            ],
        ),
    ],
)
def test_publish_unhealthy_dynamic_node_events(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    # Run test
    event_publisher.publish_unhealthy_dynamic_node_events(test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
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
                    "count": 7,
                    "instances": [
                        {"id": "queue1-dy-c5xlarge-2"},
                        {"id": "queue-dy-c5xlarge-1"},
                        {"id": "queue1-dy-c5xlarge-1"},
                        {"id": "queue1-dy-c4xlarge-1"},
                        {"id": "queue1-dy-c5xlarge-3"},
                        {"id": "queue2-dy-c5large-1"},
                        {"id": "queue2-dy-c5large-2"},
                    ],
                },
                {
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
                },
            ],
        ),
    ],
)
def test_publish_unhealthy_dynamic_node_action_events(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    instances_to_terminate = [node.name for node in test_nodes]
    power_down_nodes = [node.name for node in test_nodes]

    # Run test
    event_publisher.publish_unhealthy_dynamic_node_action_events(instances_to_terminate, power_down_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
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
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-2",
                        "address": "ip-2",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD+POWERING_DOWN",
                        "state": "IDLE",
                        "state-flags": ["CLOUD", "POWERING_DOWN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue-dy-c5xlarge-1",
                        "address": "ip-3",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD",
                        "state": "IDLE",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue"],
                        "queue-name": "queue",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                        "state": "MIXED",
                        "state-flags": ["CLOUD", "NOT_RESPONDING", "POWERING_UP"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c4xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "DOWN",
                        "state": "DOWN",
                        "state-flags": [],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c4xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    }
                },
            ],
        ),
    ],
)
def test_publish_unhealthy_static_node_events(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    # Run test
    event_publisher.publish_unhealthy_static_node_events(test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                StaticNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                StaticNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                StaticNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
                StaticNode("queue1-dy-c4xlarge-1", "ip-1", "hostname", "DOWN", "queue1"),
                StaticNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                    "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                StaticNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Failure when resuming nodes [root@2023-01-31T21:24:55]",
                ),
                StaticNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
                StaticNode(
                    "queue2-dy-c5large-3",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:UnauthorizedOperation)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
                StaticNode(
                    "queue2-dy-c5large-4",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InvalidBlockDeviceMapping)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
                StaticNode(
                    "queue2-dy-c5large-5",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:AccessDeniedException)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
                StaticNode(
                    "queue2-dy-c5large-6",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:VcpuLimitExceeded)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
                StaticNode(
                    "queue2-dy-c5large-8",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:VolumeLimitExceeded)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
                StaticNode(
                    "queue2-dy-c5large-8",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:InsufficientVolumeCapacity)Temporarily disabling node due to insufficient capacity "
                    "[root@2023-01-31T21:24:55]",
                ),
            ],
            [
                {
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
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-2",
                        "address": "ip-2",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD+POWERING_DOWN",
                        "state": "IDLE",
                        "state-flags": ["CLOUD", "POWERING_DOWN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue-dy-c5xlarge-1",
                        "address": "ip-3",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD",
                        "state": "IDLE",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue"],
                        "queue-name": "queue",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                        "state": "MIXED",
                        "state-flags": ["CLOUD", "NOT_RESPONDING", "POWERING_UP"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c4xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "DOWN",
                        "state": "DOWN",
                        "state-flags": [],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c4xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "UnauthorizedOperation",
                        "reason": "(Code:UnauthorizedOperation)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-4",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InvalidBlockDeviceMapping",
                        "reason": "(Code:InvalidBlockDeviceMapping)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-5",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "AccessDeniedException",
                        "reason": "(Code:AccessDeniedException)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-6",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "VcpuLimitExceeded",
                        "reason": "(Code:VcpuLimitExceeded)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-8",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "VolumeLimitExceeded",
                        "reason": "(Code:VolumeLimitExceeded)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-8",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientVolumeCapacity",
                        "reason": "(Code:InsufficientVolumeCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    }
                },
                {
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
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    },
                    "error-code": "InsufficientReservedInstanceCapacity",
                    "failure-type": "ice-failures",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    },
                    "error-code": "InsufficientHostCapacity",
                    "failure-type": "ice-failures",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    },
                    "error-code": "InsufficientHostCapacity",
                    "failure-type": "ice-failures",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "UnauthorizedOperation",
                        "reason": "(Code:UnauthorizedOperation)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    },
                    "error-code": "UnauthorizedOperation",
                    "failure-type": "iam-policy-errors",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-4",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InvalidBlockDeviceMapping",
                        "reason": "(Code:InvalidBlockDeviceMapping)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    },
                    "error-code": "InvalidBlockDeviceMapping",
                    "failure-type": "custom-ami-errors",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-5",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "AccessDeniedException",
                        "reason": "(Code:AccessDeniedException)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    },
                    "error-code": "AccessDeniedException",
                    "failure-type": "iam-policy-errors",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-6",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "VcpuLimitExceeded",
                        "reason": "(Code:VcpuLimitExceeded)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    },
                    "error-code": "VcpuLimitExceeded",
                    "failure-type": "vcpu-limit-failures",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-8",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "VolumeLimitExceeded",
                        "reason": "(Code:VolumeLimitExceeded)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    },
                    "error-code": "VolumeLimitExceeded",
                    "failure-type": "volume-limit-failures",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-8",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientVolumeCapacity",
                        "reason": "(Code:InsufficientVolumeCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    },
                    "error-code": "VolumeLimitExceeded",
                    "failure-type": "volume-limit-failures",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-8",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "VolumeLimitExceeded",
                        "reason": "(Code:VolumeLimitExceeded)Temporarily disabling node due to insufficient capacity "
                        "[root@2023-01-31T21:24:55]",
                    },
                    "error-code": "InsufficientVolumeCapacity",
                    "failure-type": "volume-limit-failures",
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-8",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientVolumeCapacity",
                        "reason": "(Code:InsufficientVolumeCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    },
                    "error-code": "InsufficientVolumeCapacity",
                    "failure-type": "volume-limit-failures",
                },
            ],
        ),
    ],
)
def test_publish_static_nodes_in_replacement(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    nodes_in_replacement = [node.name for node in test_nodes]
    failed_nodes = {}
    for node in test_nodes:
        if node.error_code:
            failed_nodes.setdefault(node.error_code, []).append(node.name)

    # Run test
    event_publisher.publish_static_nodes_in_replacement(test_nodes, nodes_in_replacement, failed_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
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
                    "count": 3,
                    "nodes": [
                        {"name": "queue1-dy-c5xlarge-3"},
                        {"name": "queue2-dy-c5large-1"},
                        {"name": "queue2-dy-c5large-2"},
                    ],
                },
                {
                    "partition": "queue1",
                    "resource": "c5xlarge",
                    "error-code": "InsufficientReservedInstanceCapacity",
                    "count": 1,
                    "nodes": [{"name": "queue1-dy-c5xlarge-3"}],
                },
                {
                    "partition": "queue2",
                    "resource": "c5large",
                    "error-code": "InsufficientHostCapacity",
                    "count": 2,
                    "nodes": [{"name": "queue2-dy-c5large-1"}, {"name": "queue2-dy-c5large-2"}],
                },
                {
                    "partition": "queue1",
                    "resource": "c5xlarge",
                    "failure-type": "ice-failures",
                    "error-code": "InsufficientReservedInstanceCapacity",
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    },
                },
                {
                    "partition": "queue2",
                    "resource": "c5large",
                    "failure-type": "ice-failures",
                    "error-code": "InsufficientHostCapacity",
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    },
                },
                {
                    "partition": "queue2",
                    "resource": "c5large",
                    "failure-type": "ice-failures",
                    "error-code": "InsufficientHostCapacity",
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    },
                },
            ],
        ),
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
                DynamicNode("queue1-dy-c4xlarge-1", "ip-1", "hostname", "DOWN", "queue1"),
                DynamicNode(
                    "queue1-dy-c5xlarge-3",
                    "nodeip",
                    "nodehostname",
                    "COMPLETING+DRAIN",
                    "queue1",
                ),
                DynamicNode(
                    "queue2-dy-c5large-1",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                ),
                DynamicNode(
                    "queue2-dy-c5large-2",
                    "nodeip",
                    "nodehostname",
                    "DOWN+CLOUD",
                    "queue2",
                    "(Code:SporksAreNeitherForksNorSpoons)Temporarily disabling node due to invalid utensil"
                    "[root@2023-01-31T21:24:55]",
                ),
            ],
            [{"count": 0, "nodes": []}],
        ),
    ],
    ids=["sample_of_errors", "no_ice_errors"],
)
def test_publish_insufficient_capacity_events(
    test_nodes,
    expected_details,
):
    def build_ice_map():
        node_map = {}
        for node in test_nodes:
            if node.is_ice():
                node_map.setdefault(node.queue_name, {}).setdefault(node.compute_resource_name, []).append(node)
        return node_map

    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))
    ice_map = build_ice_map()

    # Run test
    event_publisher.publish_insufficient_capacity_events(ice_map)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_instances, expected_details",
    [
        (
            [
                EC2Instance("id-2", "ip-2", "hostname", "some_launch_time"),
                # Setting launch time here for instance for static node to trigger replacement timeout
                EC2Instance("id-1", "ip-1", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-2", "ip-4", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-5", "ip-5", "hostname", "some_launch_time"),
                EC2Instance("id-6", "ip-6", "hostname", "some_launch_time"),
            ],
            [
                {
                    "count": 5,
                    "instances": [{"id": "id-2"}, {"id": "id-1"}, {"id": "id-2"}, {"id": "id-5"}, {"id": "id-6"}],
                },
                {
                    "instance": {
                        "id": "id-2",
                        "private-ip": "ip-2",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    }
                },
                {
                    "instance": {
                        "id": "id-1",
                        "private-ip": "ip-1",
                        "hostname": "hostname",
                        "launch-time": "2020-01-01 00:00:00",
                    }
                },
                {
                    "instance": {
                        "id": "id-2",
                        "private-ip": "ip-4",
                        "hostname": "hostname",
                        "launch-time": "2020-01-01 00:00:00",
                    }
                },
                {
                    "instance": {
                        "id": "id-5",
                        "private-ip": "ip-5",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    }
                },
                {
                    "instance": {
                        "id": "id-6",
                        "private-ip": "ip-6",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    }
                },
            ],
        ),
    ],
)
def test_publish_orphaned_instance_events(test_instances, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    instances_to_terminate = [instance.id for instance in test_instances]

    # Run test
    event_publisher.publish_orphaned_instance_events(test_instances, instances_to_terminate)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_map, expected_details",
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
                    "partition_failures": {
                        "partition-1": {"resource-1a": 3, "resource-1b": 2},
                        "partition-2": {"resource-2a": 1, "resource-2b": 7, "resource-2c": 3},
                    }
                },
                {"partition": "partition-1", "resource": "resource-1a", "count": 3},
                {"partition": "partition-1", "resource": "resource-1b", "count": 2},
                {"partition": "partition-2", "resource": "resource-2a", "count": 1},
                {"partition": "partition-2", "resource": "resource-2b", "count": 7},
                {"partition": "partition-2", "resource": "resource-2c", "count": 3},
            ],
        ),
    ],
)
def test_publish_entering_protected_mode_events(test_map, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    # Run test
    event_publisher.publish_entering_protected_mode_events(test_map)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
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
                },
                {
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
                },
                {
                    "health-check-type": "TriedToWalkInsteadOfRun",
                    "node": {
                        "name": "queue1-dy-c5xlarge-2",
                        "address": "ip-2",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD+POWERING_DOWN",
                        "state": "IDLE",
                        "state-flags": ["CLOUD", "POWERING_DOWN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    },
                },
                {
                    "health-check-type": "TriedToWalkInsteadOfRun",
                    "node": {
                        "name": "queue-dy-c5xlarge-1",
                        "address": "ip-3",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD",
                        "state": "IDLE",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue"],
                        "queue-name": "queue",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    },
                },
                {
                    "health-check-type": "TriedToWalkInsteadOfRun",
                    "node": {
                        "name": "queue1-dy-c5xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                        "state": "MIXED",
                        "state-flags": ["CLOUD", "NOT_RESPONDING", "POWERING_UP"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    },
                },
                {
                    "health-check-type": "TriedToWalkInsteadOfRun",
                    "node": {
                        "name": "queue1-dy-c4xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "DOWN",
                        "state": "DOWN",
                        "state-flags": [],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c4xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    },
                },
                {
                    "health-check-type": "TriedToWalkInsteadOfRun",
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    },
                },
                {
                    "health-check-type": "TriedToWalkInsteadOfRun",
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    },
                },
                {
                    "health-check-type": "TriedToWalkInsteadOfRun",
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    },
                },
            ],
        ),
    ],
)
def test_publish_nodes_failing_health_check_events(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    # Run test
    event_publisher.publish_nodes_failing_health_check_events("TriedToWalkInsteadOfRun", test_nodes, test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_nodes, expected_details",
    [
        (
            [
                DynamicNode(
                    "queue1-dy-c5xlarge-2", "ip-2", "hostname", "IDLE+CLOUD+POWERING_DOWN", "queue1"
                ),  # powering_down
                DynamicNode("queue-dy-c5xlarge-1", "ip-3", "hostname", "IDLE+CLOUD", "queue"),
                DynamicNode(
                    "queue1-dy-c5xlarge-1", "ip-1", "hostname", "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP", "queue1"
                ),  # bootstrap failure dynamic
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
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-2",
                        "address": "ip-2",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD+POWERING_DOWN",
                        "state": "IDLE",
                        "state-flags": ["CLOUD", "POWERING_DOWN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue-dy-c5xlarge-1",
                        "address": "ip-3",
                        "hostname": "hostname",
                        "state-string": "IDLE+CLOUD",
                        "state": "IDLE",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue"],
                        "queue-name": "queue",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "MIXED+CLOUD+NOT_RESPONDING+POWERING_UP",
                        "state": "MIXED",
                        "state-flags": ["CLOUD", "NOT_RESPONDING", "POWERING_UP"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c4xlarge-1",
                        "address": "ip-1",
                        "hostname": "hostname",
                        "state-string": "DOWN",
                        "state": "DOWN",
                        "state-flags": [],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c4xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": None,
                        "reason": None,
                    }
                },
                {
                    "node": {
                        "name": "queue1-dy-c5xlarge-3",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "COMPLETING+DRAIN",
                        "state": "COMPLETING",
                        "state-flags": ["DRAIN"],
                        "partitions": ["queue1"],
                        "queue-name": "queue1",
                        "compute-resource": "c5xlarge",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": True,
                        "error-code": "InsufficientReservedInstanceCapacity",
                        "reason": "(Code:InsufficientReservedInstanceCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-1",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Failure when resuming nodes "
                        "[root@2023-01-31T21:24:55]",
                    }
                },
                {
                    "node": {
                        "name": "queue2-dy-c5large-2",
                        "address": "nodeip",
                        "hostname": "nodehostname",
                        "state-string": "DOWN+CLOUD",
                        "state": "DOWN",
                        "state-flags": ["CLOUD"],
                        "partitions": ["queue2"],
                        "queue-name": "queue2",
                        "compute-resource": "c5large",
                        "node-type": "dy",
                        "instance": "None",
                        "slurmd-start-time": None,
                        "up-time": 0,
                        "idle-time": 0,
                        "is-running-job": False,
                        "error-code": "InsufficientHostCapacity",
                        "reason": "(Code:InsufficientHostCapacity)Temporarily disabling node due to insufficient "
                        "capacity [root@2023-01-31T21:24:55]",
                    }
                },
            ],
        ),
    ],
)
def test_publish_failed_health_check_nodes_in_replacement(test_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    # Run test
    event_publisher.publish_failed_health_check_nodes_in_replacement(test_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)


@pytest.mark.parametrize(
    "test_instances, failed_nodes, expected_details",
    [
        (
            [
                EC2Instance("id-2", "ip-2", "hostname", "some_launch_time"),
                # Setting launch time here for instance for static node to trigger replacement timeout
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
                    "count": 5,
                    "nodes": [
                        {"name": "node-id-2", "id": "id-2"},
                        {"name": "node-id-1", "id": "id-1"},
                        {"name": "node-id-2", "id": "id-2"},
                        {"name": "node-id-5", "id": "id-5"},
                        {"name": "node-id-6", "id": "id-6"},
                    ],
                },
                {
                    "node": {"name": "node-id-2"},
                    "instance": {
                        "id": "id-2",
                        "private-ip": "ip-2",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    },
                },
                {
                    "node": {"name": "node-id-1"},
                    "instance": {
                        "id": "id-1",
                        "private-ip": "ip-1",
                        "hostname": "hostname",
                        "launch-time": "2020-01-01 00:00:00",
                    },
                },
                {
                    "node": {"name": "node-id-2"},
                    "instance": {
                        "id": "id-2",
                        "private-ip": "ip-4",
                        "hostname": "hostname",
                        "launch-time": "2020-01-01 00:00:00",
                    },
                },
                {
                    "node": {"name": "node-id-5"},
                    "instance": {
                        "id": "id-5",
                        "private-ip": "ip-5",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    },
                },
                {
                    "node": {"name": "node-id-6"},
                    "instance": {
                        "id": "id-6",
                        "private-ip": "ip-6",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    },
                },
                {
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
                },
                {"error-code": "Error1", "failure-type": "other-failures", "node": {"name": "node-a-1"}},
                {"error-code": "Error1", "failure-type": "other-failures", "node": {"name": "node-a-2"}},
                {"error-code": "Error1", "failure-type": "other-failures", "node": {"name": "node-a-3"}},
                {"error-code": "Error2", "failure-type": "other-failures", "node": {"name": "node-b-1"}},
                {"error-code": "Error2", "failure-type": "other-failures", "node": {"name": "node-b-2"}},
                {
                    "error-code": "InsufficientInstanceCapacity",
                    "failure-type": "ice-failures",
                    "node": {"name": "ice-a-1"},
                },
                {
                    "error-code": "InsufficientInstanceCapacity",
                    "failure-type": "ice-failures",
                    "node": {"name": "ice-a-2"},
                },
                {
                    "error-code": "InsufficientInstanceCapacity",
                    "failure-type": "ice-failures",
                    "node": {"name": "ice-a-3"},
                },
                {"error-code": "InsufficientHostCapacity", "failure-type": "ice-failures", "node": {"name": "ice-b-1"}},
                {"error-code": "InsufficientHostCapacity", "failure-type": "ice-failures", "node": {"name": "ice-b-2"}},
                {
                    "error-code": "InsufficientReservedInstanceCapacity",
                    "failure-type": "ice-failures",
                    "node": {"name": "ice-c-1"},
                },
                {
                    "error-code": "InsufficientReservedInstanceCapacity",
                    "failure-type": "ice-failures",
                    "node": {"name": "ice-c-2"},
                },
                {
                    "error-code": "InsufficientReservedInstanceCapacity",
                    "failure-type": "ice-failures",
                    "node": {"name": "ice-c-3"},
                },
                {
                    "error-code": "MaxSpotInstanceCountExceeded",
                    "failure-type": "ice-failures",
                    "node": {"name": "ice-d-1"},
                },
                {
                    "error-code": "MaxSpotInstanceCountExceeded",
                    "failure-type": "ice-failures",
                    "node": {"name": "ice-d-2"},
                },
                {"error-code": "Unsupported", "failure-type": "ice-failures", "node": {"name": "ice-e-1"}},
                {"error-code": "Unsupported", "failure-type": "ice-failures", "node": {"name": "ice-e-2"}},
                {"error-code": "Unsupported", "failure-type": "ice-failures", "node": {"name": "ice-e-3"}},
                {"error-code": "SpotMaxPriceTooLow", "failure-type": "ice-failures", "node": {"name": "ice-f-1"}},
                {"error-code": "SpotMaxPriceTooLow", "failure-type": "ice-failures", "node": {"name": "ice-f-2"}},
                {
                    "error-code": "VcpuLimitExceeded",
                    "failure-type": "vcpu-limit-failures",
                    "node": {"name": "vcpu-g-1"},
                },
                {
                    "error-code": "VolumeLimitExceeded",
                    "failure-type": "volume-limit-failures",
                    "node": {"name": "vle-h-1"},
                },
                {
                    "error-code": "VolumeLimitExceeded",
                    "failure-type": "volume-limit-failures",
                    "node": {"name": "vle-h-2"},
                },
                {
                    "error-code": "InsufficientVolumeCapacity",
                    "failure-type": "volume-limit-failures",
                    "node": {"name": "ivc-i-1"},
                },
                {
                    "error-code": "InsufficientVolumeCapacity",
                    "failure-type": "volume-limit-failures",
                    "node": {"name": "ivc-i-2"},
                },
                {
                    "error-code": "InsufficientVolumeCapacity",
                    "failure-type": "volume-limit-failures",
                    "node": {"name": "ivc-i-3"},
                },
                {
                    "error-code": "InvalidBlockDeviceMapping",
                    "failure-type": "custom-ami-errors",
                    "node": {"name": "ibdm-j-1"},
                },
                {
                    "error-code": "InvalidBlockDeviceMapping",
                    "failure-type": "custom-ami-errors",
                    "node": {"name": "ibdm-j-2"},
                },
                {
                    "error-code": "InvalidBlockDeviceMapping",
                    "failure-type": "custom-ami-errors",
                    "node": {"name": "ibdm-j-3"},
                },
                {
                    "error-code": "UnauthorizedOperation",
                    "failure-type": "iam-policy-errors",
                    "node": {"name": "iam-k-1"},
                },
                {
                    "error-code": "UnauthorizedOperation",
                    "failure-type": "iam-policy-errors",
                    "node": {"name": "iam-k-2"},
                },
                {
                    "error-code": "AccessDeniedException",
                    "failure-type": "iam-policy-errors",
                    "node": {"name": "iam-l-1"},
                },
            ],
        ),
        (
            [
                EC2Instance("id-2", "ip-2", "hostname", "some_launch_time"),
                # Setting launch time here for instance for static node to trigger replacement timeout
                EC2Instance("id-1", "ip-1", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-2", "ip-4", "hostname", datetime(2020, 1, 1, 0, 0, 0)),
                EC2Instance("id-5", "ip-5", "hostname", "some_launch_time"),
                EC2Instance("id-6", "ip-6", "hostname", "some_launch_time"),
            ],
            {},
            [
                {
                    "count": 5,
                    "nodes": [
                        {"name": "node-id-2", "id": "id-2"},
                        {"name": "node-id-1", "id": "id-1"},
                        {"name": "node-id-2", "id": "id-2"},
                        {"name": "node-id-5", "id": "id-5"},
                        {"name": "node-id-6", "id": "id-6"},
                    ],
                },
                {
                    "node": {"name": "node-id-2"},
                    "instance": {
                        "id": "id-2",
                        "private-ip": "ip-2",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    },
                },
                {
                    "node": {"name": "node-id-1"},
                    "instance": {
                        "id": "id-1",
                        "private-ip": "ip-1",
                        "hostname": "hostname",
                        "launch-time": "2020-01-01 00:00:00",
                    },
                },
                {
                    "node": {"name": "node-id-2"},
                    "instance": {
                        "id": "id-2",
                        "private-ip": "ip-4",
                        "hostname": "hostname",
                        "launch-time": "2020-01-01 00:00:00",
                    },
                },
                {
                    "node": {"name": "node-id-5"},
                    "instance": {
                        "id": "id-5",
                        "private-ip": "ip-5",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    },
                },
                {
                    "node": {"name": "node-id-6"},
                    "instance": {
                        "id": "id-6",
                        "private-ip": "ip-6",
                        "hostname": "hostname",
                        "launch-time": "some_launch_time",
                    },
                },
                {
                    "other-failures": {"count": 0},
                    "ice-failures": {"count": 0},
                    "vcpu-limit-failures": {"count": 0},
                    "volume-limit-failures": {"count": 0},
                    "custom-ami-errors": {"count": 0},
                    "iam-policy-errors": {"count": 0},
                    "total": 0,
                },
            ],
        ),
    ],
)
def test_publish_node_launch_events(test_instances, failed_nodes, expected_details):
    received_events = []
    event_publisher = ClusterEventPublisher(event_handler(received_events))

    successful_nodes = [(f"node-{instance.id}", instance) for instance in test_instances]

    # Run test
    event_publisher.publish_node_launch_events(successful_nodes, failed_nodes)

    # Assert calls
    assert_that(received_events).is_length(len(expected_details))
    for received_event, expected_detail in zip(received_events, expected_details):
        assert_that(received_event).is_equal_to(expected_detail)
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with
# the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
import os
from configparser import ConfigParser
from datetime import datetime, timezone
from logging.config import fileConfig

from slurm_plugin.common import is_clustermgtd_heartbeat_valid, metric_publisher, metric_publisher_noop
from slurm_plugin.slurm_resources import CONFIG_FILE_DIR

log = logging.getLogger(__name__)
metrics_logger = log.getChild("metrics")

_publish_metric = metric_publisher_noop


class SlurmSuspendConfig:
    DEFAULTS = {
        "clustermgtd_timeout": 300,
        "logging_config": os.path.join(os.path.dirname(__file__), "logging", "parallelcluster_suspend_logging.conf"),
    }

    def __init__(self, config_file_path):
        config = ConfigParser()
        try:
            config.read_file(open(config_file_path, "r"))
        except IOError:
            log.error("Cannot read slurm cloud bursting scripts configuration file: %s", config_file_path)
            raise

        self.clustermgtd_timeout = config.getint(
            "slurm_suspend",
            "clustermgtd_timeout",
            fallback=self.DEFAULTS.get("clustermgtd_timeout"),
        )
        self.clustermgtd_heartbeat_file_path = config.get("slurm_suspend", "clustermgtd_heartbeat_file_path")
        self.logging_config = config.get(
            "slurm_suspend", "logging_config", fallback=self.DEFAULTS.get("logging_config")
        )
        self.cluster_name = config.get("slurm_suspend", "cluster_name", fallback="unknown")
        self.head_node_instance_id = config.get("slurm_suspend", "instance_id", fallback="unknown")
        log.info(self.__repr__())


def main():
    default_log_file = "/var/log/parallelcluster/slurm_suspend.log"
    logging.basicConfig(
        filename=default_log_file,
        level=logging.INFO,
        format="%(asctime)s - [%(name)s:%(funcName)s] - %(levelname)s - %(message)s",
    )
    log.info("SuspendProgram startup.")
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", help="Nodes to release")
    args = parser.parse_args()
    config_file = os.environ.get("CONFIG_FILE", os.path.join(CONFIG_FILE_DIR, "parallelcluster_slurm_suspend.conf"))
    suspend_config = SlurmSuspendConfig(config_file)
    try:
        # Configure root logger
        fileConfig(suspend_config.logging_config, disable_existing_loggers=False)
    except Exception as e:
        log.warning(
            "Unable to configure logging from %s, using default settings and writing to %s.\nException: %s",
            suspend_config.logging_config,
            default_log_file,
            e,
        )

    global _publish_metric
    _publish_metric = metric_publisher(
        metrics_logger, suspend_config.cluster_name, "slurm_suspend", suspend_config.head_node_instance_id
    )

    log.info("Suspending following nodes. Clustermgtd will cleanup orphaned instances: %s", args.nodes)
    current_time = datetime.now(tz=timezone.utc)
    if not is_clustermgtd_heartbeat_valid(
        current_time, suspend_config.clustermgtd_timeout, suspend_config.clustermgtd_heartbeat_file_path
    ):
        error_message = "No valid clustermgtd heartbeat detected, clustermgtd is down! "
        "Please check clustermgtd log for error.\n"
        "Nodes will be reset to POWER_SAVE state after SuspendTimeout. "
        "The backing EC2 instances may not be correctly terminated.\n"
        "Please check and terminate any orphaned instances in EC2!"
        log.error(error_message)
        _publish_metric("ERROR", error_message, "suspend_error", nodes=args.nodes)
    else:
        log.info("SuspendProgram finished. Nodes will be available after SuspendTimeout")
        _publish_metric("INFO", "Node Suspended", "suspend_node", nodes=args.nodes)


if __name__ == "__main__":
    main()

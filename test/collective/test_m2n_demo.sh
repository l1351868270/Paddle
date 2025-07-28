# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F"=" '{print $1}'`; do
unset ${name}
done

export LD_LIBRARY_PATH=/ssd2/lishuliang/nvshmem/lib:${LD_LIBRARY_PATH}
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0
# export IP_LIST="10.94.130.150,10.94.130.151,10.94.130.152"
export IP_LIST="10.94.130.151,10.94.130.152,10.94.130.153"
export NCCL_DEBUG=WARN

export devices=0,1,2,3,4,5,6,7
export start_port=6073
python3.10 -m paddle.distributed.launch \
        --gpus ${devices} \
        --ips ${IP_LIST} \
        --start_port ${start_port} \
        test_m2n_demo.py

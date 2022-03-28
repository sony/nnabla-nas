# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

NNABLA_NAS_BUILD_ENV_DOCKER_IMAGE_NAME=nnabla-nas-build

.PHONY:nnabla-nas-build-env
nnabla-nas-build-env:
	docker build -f docker/development/Dockerfile.build -t $(NNABLA_NAS_BUILD_ENV_DOCKER_IMAGE_NAME) .

.PHONY:nnabla-nas-wheel
nnabla-nas-wheel:
	python3 setup.py bdist_wheel

.PHONY:bwd-nnabla-nas-wheel
bwd-nnabla-nas-wheel:nnabla-nas-build-env
	docker run -v $$(pwd):$$(pwd) -w $$(pwd) -u $$(id -u):$$(id -g) $(NNABLA_NAS_BUILD_ENV_DOCKER_IMAGE_NAME) make nnabla-nas-wheel


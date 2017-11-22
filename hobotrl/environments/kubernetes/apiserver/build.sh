#!/usr/bin/env bash
#
# usage: ./build.sh 0.1.5

version=$1
image=$(docker build -f Dockerfile.server . | tail -n 2 | grep "Successfully built" | awk '{print $3;}')
tag="docker.hobot.cc/carsim/api:$version"
echo "tagging $image to $tag"
docker tag $image $tag
docker push $tag  

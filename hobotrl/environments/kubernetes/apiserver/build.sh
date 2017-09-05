#
# usage: ./build.sh 0.1.2

version=$1
image=$(docker build -f Dockerfile.server . | grep "Successfully built" | awk '{print $3;}')
tag="docker.hobot.cc/carsim/api:$version"
echo "tagging $image to $tag"
docker tag $image $tag
docker push $tag  

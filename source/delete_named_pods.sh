#!/usr/bin/env bash

#usage bash delete_named_pods queryname
queryname=$1
podlist=$(kubectl get pods --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}' | grep $queryname)

for name in ${podlist[@]}
    do
    echo $name
    kubectl delete pod $name
    done

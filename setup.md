## On laptop - server:

`flower-superlink --insecure`

## On pis:

### On pi1:
`flower-supernode --insecure --superlink="192.168.1.121:9092" --node-config="dataset-path='/home/admin/fed_learning/fashionmnist_part_1'" `

### On pi2:
`flower-supernode --insecure --superlink="192.168.1.121:9092" --node-config="dataset-path='/home/admin/fed_learning/fashionmnist_part_2'" ` 

## Run the app on laptop:
`flwr run . embedded-federation --stream`

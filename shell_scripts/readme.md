
Useful commands for running the shell scripts in the background, without ssh tether: 

nohup ./train.sh > nohup_train_v5_100epochs.out 2>&1 & 

and to generate new graphs: 

nohup ./generate_hact_graph_test.sh > nohup_graph-gen_test_v5.out 2>&1 & 
nohup ./generate_hact_graph_train.sh > nohup_graph-gen_train_v5.out 2>&1 & 
nohup ./generate_hact_graph_val.sh > nohup_graph-gen_val_v5.out 2>&1 & 


docker kill rapp-ddqn
# docker rmi rapp-ddqn:1.0.0
# docker build -t rapp-ddqn:1.0.0 .
sudo docker run -v $(pwd)/main.py:/app/main.py -v $(pwd)/model_picture/:/app/picture/ --name rapp-ddqn rapp-ddqn:1.0.0 
#tail -f /dev/null
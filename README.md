# Intro to GNNs
For getting more practice with graph neural networks

# Citations

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/index.html
- Revisiting Semi-Supervised Learning with Graph Embeddings: https://arxiv.org/abs/1603.08861



# Installation Instructions
As of March 2024, pytorch-geometric does not support Windows or M-silicon Macs. Therefor it is recommended to use a Linux machine or WSL for this project. 

## Using Anaconda

1. Create your virtual environment
    
    ```conda create -n <env_name> python==3.11```

2. Activate your virtual environment

    ```conda activate <env_name>```

3. Install the requirements with pip

    ```pip install -r requirements.txt```

4. Deactivate when you're done

    ```conda deactivate```

## Using Python Virtual Environment

1. Create your virtual environment

    ```python -m venv path/to/env/env_name```

2. Activate your virtual environment

    ```source path/to/env/env_name/bin/activate```

3. Upgrade your pip to the latest version

    ```pip install --upgrade pip```

4. Install the requirements with pip

    ```pip install -r requirements.txt```

4. Deactivate when you're done

    ```deactivate```
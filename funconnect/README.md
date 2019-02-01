# Getting started

This assumes you have `docker` and `docker-compose` installed. 

Create and `.env` file in this directory with your datajoint credentials. 

```bash
DJ_HOST=10.28.0.34
DJ_USER=<your database user>
DJ_PASS=<your database password>
```

To start the notebook, you simpley need

```bash
sudo docker-compose up notebook
```

If you want to run it in the background use

```bash
sudo docker-compose up -d notebook
```

Now use your browser and navigate to 

```
http://YOURCOMPUTER:8888
```

Notebooks are mounted to the local `notebooks` folder. If you want to change that or the port, change the `docker-compose.yml` file. 

# Custom settings

If you want to develop outside the container it might be useful to mount the outside code in. 
You can do that by installing the package outside with 
```bash
pip3 install -e funconnect
```

Then check the `docker-compose.yml` for the `custom` service and how that directory is mounted into the container. 
You need to change the location of the to the left of `:` to where it is on your computer. You can also mount in other directories. 
For instance, we might want to have a shared Dropbox directory for figures, that can be mounted there. 

# Before you commit to git

Install `nbstripout` to clean notebooks

```bash
pip3 install nbstripout
```

and in the local directory

```bash
nbstripout --install
```

You only need to do that once. 
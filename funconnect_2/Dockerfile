#FROM datajoint/datajoint:latest
FROM ninai/pipeline:base
LABEL maintainer="Edgar Y. Walker, Fabian Sinz, Erick Cobos"

RUN apt-get -y update && \
    apt-get -y install graphviz libxml2-dev python3-cairosvg parallel

RUN pip3 install datajoint --upgrade
RUN pip3 install python-igraph xlrd

WORKDIR /src

RUN pip3 install ipyvolume jupyterlab statsmodels pycircstat nose
RUN pip3 install seaborn --upgrade
RUN pip3 install jgraph

ADD . /src/funconnect
RUN pip3 install -e /src/funconnect/python

WORKDIR /notebooks

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter.sh"]

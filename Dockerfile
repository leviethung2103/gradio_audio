# Python 3.11
# FROM jupyter/scipy-notebook:2023-08-19
# Python 3.7 - Use pytorch version 1.13 -> Whisper
# FROM jupyter/scipy-notebook:1aac87eb7fa5 
# Python 3.8 - Use pytorch version 1.13 -> Whisper
# FROM jupyter/scipy-notebook:a374cab4fcb6
# Python 3.x ? - Use Platform 
FROM jupyter/scipy-notebook:x86_64-ubuntu-20.04
USER root
WORKDIR /home/jovyan/work
ARG DEBIAN_FRONTEND=noninteractive

# RUN apt-get update -qqy \
#   && apt-get -qqy install \
#        dumb-init gnupg wget ca-certificates apt-transport-https \
#        ttf-wqy-zenhei \
#   && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
#   && echo "deb https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
#   && apt-get update -qqy \
#   && apt-get -qqy install google-chrome-stable \
#   && rm /etc/apt/sources.list.d/google-chrome.list \
#   && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# install pyodbc -mysql
RUN apt-get update && apt-get install unixodbc -y

COPY ./requirements.txt /home/jovyan/work/requirements.txt

RUN pip install -r requirements.txt

CMD ["start-notebook.sh", "--NotebookApp.token=", "--NotebookApp.password="]

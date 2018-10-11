# use miniconda base image
FROM continuumio/miniconda

# Set the ENTRYPOINT to use bash
ENTRYPOINT [ “/bin/bash”, “-c” ]

EXPOSE 5000

# set up work directory.
WORKDIR /home/me/dev/

# copy necessary project files.
COPY ./*.* ./

# create new conda env and install packages.
#RUN conda create -n kashir_cheez python=3.6
RUN [ "/bin/bash", "-c", "conda create -n kashir_cheez python=3.6" ]
#RUN conda activate kashir_cheez
RUN [ "/bin/bash", "-c", "source activate kashir_cheez" ]
RUN [ "/bin/bash", "-c", "pip install -r requirements.txt" ]

# run app
CMD python app.py
#!/bin/bash
curl -L -o blackadder.zip\
  https://www.kaggle.com/api/v1/datasets/download/soumee2000/blackadderfullscriptsrowan-atkinson

unzip blackadder.zip
rm blackadder.zip
mv Season\ 1.txt season_1.txt
mv Season\ 2.txt season_2.txt
mv Season\ 3.txt season_3.txt
mv Season\ 4.txt season_4.txt
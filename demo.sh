#!/bin/sh

python main.py --mode=data_creation
python main.py --mode=training
python main.py --mode=prediction --audio_input_prediction=hydrophone_sensor_2020-08-28_10_25_22_in_the_tube_point_comma.595226_none.wav --audio_output_prediction=hydrophone_sensor_2020-08-28_10_25_22_in_the_tube_point_comma.595226_none_out.wav --name_model=model_best
python main.py --mode=prediction --audio_input_prediction=hydrophone_sensor_2020-08-31_15_07_41.159862_100_lightness_point_none.wav --audio_output_prediction=hydrophone_sensor_2020-08-31_15_07_41.159862_100_lightness_point_none_out.wav --name_model=model_best
python main.py --mode=prediction --audio_input_prediction=hydrophone_sensor_2020-08-31_15_08_35.565245_20_lightness_point_none.wav --audio_output_prediction=hydrophone_sensor_2020-08-31_15_08_35.565245_20_lightness_point_none_out.wav --name_model=model_best

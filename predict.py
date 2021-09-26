import os

from train import config
from unet3d.prediction import run_validation_cases

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config["data_file"] = os.path.abspath("cornary_altery_data.h5")
config["model_file"] = os.path.abspath("model.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")
#config["validation_file"]=config["training_file"]
config["labels"] = ([1])
config["all_mod#alities"] = ["contrast_split", "contrast_texture_feature_0", "contrast_texture_feature_1", "contrast_texture_feature_2"]
config["training_modalities"] = config["all_modalities"]
def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,
                         overlap=0)


if __name__ == "__main__":
    main()

import os
import glob

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import isensee2017_model
from unet3d.model.twolevelmodel import twolevel_model
from unet3d.training import load_old_model, train_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = dict()
config["image_shape"] = (256, 256, 256)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (128, 128, 128)  # switch to None to train on the whole image
config["labels"] = ([1])  # the label numbers on the input image
config["n_base_filters"] =16
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["contrast_split", "contrast_texture_feature_0", "contrast_texture_feature_1", "contrast_texture_feature_2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = False  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 8  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 20  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4#5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.95  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = None  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("cornary_altery_data.h5")
config["model_file"] = os.path.abspath("model.h5")
#config["model_file"] = os.path.abspath("2017_model.h5")
#config["training_file"] = os.path.abspath("validation_ids.pkl")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")
#config["overwrite"] = True  # If True, will previous files. If False, will use previously written files.
config["overwrite"] = False
config["two_level"] = False
config["two_level_file"] = os.path.abspath("twolevel_model.h5")
def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["cor_label"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False,twolevel=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids)
    data_file_opened = open_data_file(config["data_file"])
    if not overwrite and os.path.exists(config["two_level_file"])  and twolevel:
        model = load_old_model(config["two_level_file"],config["initial_learning_rate"])

    elif not overwrite and os.path.exists(config["model_file"]) and twolevel:
        head_model = load_old_model(config["model_file"])
        config["model_file"] = config["two_level_file"]
        model = twolevel_model(head_model,input_shape=config["input_shape"], n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"])

    elif not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"],config["initial_learning_rate"])
        #model.summary()
        #exit()
    else:
        # instantiate new model
        model = isensee2017_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"])

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                #learning_rate_epochs=5,
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"],twolevel=config["two_level"])


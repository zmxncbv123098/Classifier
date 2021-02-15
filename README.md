# Classifier Study

Python simple classifier. 

### Requirements
1. Install PyTorch from http://pytorch.org
2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```
> requirements.txt was created by the command ```pip freeze > requirements.txt ```

### Download Data
- Download Coco annotations from [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- Run `coco_dataset_download.py`
- The folder *category_imgs* should be in the same folder with `main.py`

### Training and validating your model
Run the script `main.py` to train your model.

Modify `main.py`, `model.py` and `dataset.py` for your assignment, with an aim to make the validation score better.

- By default the images are loaded and resized to 32x32 pixels and normalized to zero-mean and standard deviation of 1. See `dataset.py` for the `get_transforms`.
- By default a validation set is split for you from the training set. See `main.py` on how this is done.


### License

zmxncbv123098verriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be

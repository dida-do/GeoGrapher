Basic cutting of datasets
#########################

.. todo::

    - other options
    - saving and loading etc

Cutting every image to a grid of images
=======================================

Cutting every image in the dataset to a grid of images::


    from rs_tools.cutters import DSCutterEveryImgToGrid
    cutter = DSCutterEveryImgToGrid(
        new_img_size=512,
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>)
    cutter.cut()

Updating the ``target_data_dir`` after the ``source_data_dir`` has
grown since it was cut::

    cutter.update()

Basic cutting: cutting images around vector features
====================================================

Cutting images around vector features (e.g. create 512 x 512 pixel
cutouts around vector features from 10980 x 10980 Sentinel-2 tiles)::

    from rs_tools.cutters import DSCutterImgsAroundEveryFeature
    cutter = DSCutterImgsAroundEveryFeature(
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>
        new_img_size: Optional[ImgSize]
        new_img_size=512,
        target_img_count=2,
        mode: "random")
    cutter.cut()

Updating the ``target_data_dir`` after the ``source_data_dir``
has grown (more images or vector features) since it was cut::

    cutter.update()

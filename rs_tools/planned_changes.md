
planned design changes:

associator:
* I only defined errors for the download method of the associator and generally I think the error handling could be improved. 
* make the associator remember not just the containment relation ('contains' or 'intersects') between images and polygons, but also which images were downloaded for which polygons. this tells us which images to stitch together if we've downloaded several images for a polygon that need stitching together. could be easily done by changing the edge_data in the graph, probably to a dict, and then modifying the functions/methods in the class that deal with the edge_data appropriately. it's probably best to just add parameters to the methods that return information about the graph structure (polygons_intersecting_img etc.) to allow us to query the associator about this information, but I'm not sure. we might also want to just change the interface in this case. we'll definitely want a imgs_intersecting_polygon method we can use when stitching together images.
* right now, the associator stores exceptions that occur when downloading an image in the polygons_df (geo)dataframe. it would make more sense to store the exceptions in the imgs_df, especially because the associator can now deal with the case where several images have to be downloaded for a single polygon (if no image exists that fully contains it and we have to stitch together an image). for this we have to allow entries in the imgs_df dataframe without a geometry.
* error handling in the download_missing_imgs_for_polygons_df: currently, if an exception is thrown when downloading images for a polygon (e.g. a download error for an image) none of the downloaded images (if there are any) will be integrated into the associator. there might still be images that were successfully downloaded though and it would make sense to integrate those. 
* combine __download_imgs_for_polygon__ and __process_downloaded_img_file__ functions?
* probably don't need __download_imgs_for_polygon__ function to return polygon_info_dict anymore, right now it's still there for backward compatibility.
* maybe combine have_img and have_img_downloaded fields in imgs_df?

graph:
* make it possible not to have to specify the vertex color where it can be inferred.


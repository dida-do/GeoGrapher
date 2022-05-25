import json
import os
import sys
import time
from pathlib import Path

import fiona
import geopandas as gpd
import pyproj
import rasterio as rio
import requests


class GeoserverAPI:

    def __init__(self, host, directory: Path, layer_type: str, workspace: str):
        """Class for adding a new shapefile or raster file to geoserver.

        This includes creating a new datastore and layer.
        Parameters:
                host: host of the geoserver, e.g. localhost:8080
                layer_type: type of layer to be created, either vector or raster data
                file: directory + filename of the vector or raster file
                workspace: name of the workspace the layer should belong to
        """
        self.auth = ('username', 'password')
        self.host = host
        self.layer_type = layer_type
        self.workspace = workspace
        self.directory = directory
        # initialize names for datastore and layer and file
        self.datastore = None
        self.name = None
        self.file = None

        if not self.__check(to_check='workspace'):
            self.create_workspace()

    def __check(self, to_check: str):

        if to_check == 'workspace':
            response = requests.get(
                f'https://{self.host}/geoserver/rest/workspaces',
                auth=self.auth,
                headers={'Content-type': 'application/json'})
            try:
                workspaces = [
                    ws['name']
                    for ws in response.json()['workspaces']['workspace']
                ]
            except:
                return False

            if self.workspace in workspaces:
                return True
            else:
                return False

        elif to_check == 'datastore':
            response = requests.get(
                f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/datastores',
                auth=self.auth,
                headers={'Content-type': 'application/json'})

            try:
                datastores = [
                    ds['name']
                    for ds in response.json()['dataStores']['dataStore']
                ]
            except:
                return False

            if self.datastore in datastores:
                return True
            else:
                return False

        elif to_check == 'coveragestore':
            response = requests.get(
                f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/coveragestores',
                auth=self.auth,
                headers={'Content-type': 'application/json'})

            try:
                coveragestores = [
                    cs['name'] for cs in response.json()['coverageStores']
                    ['coverageStore']
                ]
            except:
                return False

            if self.datastore in coveragestores:
                return True
            else:
                return False

    def __create_layer(self):

        if self.layer_type == 'vector':
            if self.__check(to_check='datastore'):
                return print(f'Datastore {self.datastore} already exists.')
            self.__create_datastore()
            time.sleep(1)
            self.__create_featuretype()
        elif self.layer_type == 'raster':
            if self.__check(to_check='coveragestore'):
                return print(f'Datastore {self.datastore} already exists.')
            self.__create_coveragestore()
            time.sleep(1)
            self.__create_coverage()

    def __create_datastore(self):

        body = {
            'dataStore': {
                'name': self.datastore,
                'description': '',
                'type': 'Shapefile',
                'enabled': True,
                'workspace': {
                    'name':
                    self.workspace,
                    'href':
                    f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}.json'
                },
                'connectionParameters': {
                    'entry': [{
                        '@key': 'charset',
                        '$': 'ISO-8859-1'
                    }, {
                        '@key': 'filetype',
                        '$': 'shapefile'
                    }, {
                        '@key': 'create spatial index',
                        '$': 'true'
                    }, {
                        '@key': 'memory mapped buffer',
                        '$': 'false'
                    }, {
                        '@key': 'timezone',
                        '$': 'Etc/UTC'
                    }, {
                        '@key': 'enable spatial index',
                        '$': 'true'
                    }, {
                        '@key': 'namespace',
                        '$': f'http://{self.workspace}'
                    }, {
                        '@key': 'cache and reuse memory maps',
                        '$': 'true'
                    }, {
                        '@key': 'url',
                        '$': f'file:{self.file}'
                    }, {
                        '@key': 'fstype',
                        '$': 'shape'
                    }]
                },
                '_default': False
            }
        }

        response = requests.post(
            f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/datastores',
            auth=self.auth,
            headers={'Content-type': 'application/json'},
            data=json.dumps(body))

        if response.status_code == 201:
            return print('Adding new datastore was successful')
        else:
            raise GeoserverAPIError(
                f'{response.status_code}:{response.content}')

    def __create_featuretype(self):

        vector_data = self.file
        with fiona.open(vector_data) as src:
            bounds = src.bounds

        body = {
            'featureType': {
                'name':
                self.name,
                'nativeName':
                self.datastore,
                'namespace': {
                    'name':
                    self.workspace,
                    'href':
                    f'https://{self.host}/geoserver/rest/namespaces/{self.workspace}.json'
                },
                'title':
                self.name,
                'keywords': {
                    'string': ['features', self.datastore]
                },
                'nativeCRS':
                'GEOGCS["WGS 84", DATUM["World Geodetic System 1984", SPHEROID["WGS 84", 6378137.0, 298.257223563, AUTHORITY["EPSG","7030"]],'
                'AUTHORITY["EPSG","6326"]], PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]], UNIT["degree", 0.017453292519943295],'
                'AXIS["Geodetic longitude", EAST], AXIS["Geodetic latitude", NORTH], AUTHORITY["EPSG","4326"]]',
                'srs':
                'EPSG:4326',
                'nativeBoundingBox': {
                    'minx': bounds[0],
                    'maxx': bounds[2],
                    'miny': bounds[1],
                    'maxy': bounds[3],
                    'crs': 'EPSG:4326'
                },
                'latLonBoundingBox': {
                    'minx': bounds[0],
                    'maxx': bounds[2],
                    'miny': bounds[1],
                    'maxy': bounds[3],
                    'crs': 'EPSG:4326'
                },
                'projectionPolicy':
                'FORCE_DECLARED',
                'enabled':
                True,
                'metadata': {
                    'entry': {
                        '@key': 'cachingEnabled',
                        '$': 'false'
                    }
                },
                'store': {
                    '@class':
                    'dataStore',
                    'name':
                    f'{self.workspace}:{self.datastore}',
                    'href':
                    f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/datastores/{self.datastore}.json'
                },
                'serviceConfiguration':
                False,
                'simpleConversionEnabled':
                False,
                'maxFeatures':
                0,
                'numDecimals':
                0,
                'padWithZeros':
                False,
                'forcedDecimal':
                False,
                'overridingServiceSRS':
                False,
                'skipNumberMatched':
                False,
                'circularArcPresent':
                False,
                'attributes': {
                    'attribute': [{
                        'name':
                        'the_geom',
                        'minOccurs':
                        0,
                        'maxOccurs':
                        1,
                        'nillable':
                        True,
                        'binding':
                        'org.locationtech.jts.geom.MultiPolygon'
                    }, {
                        'name': 'class',
                        'minOccurs': 0,
                        'maxOccurs': 1,
                        'nillable': True,
                        'binding': 'java.lang.String',
                        'length': 80
                    }]
                }
            }
        }

        response = requests.post(
            f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/datastores/{self.datastore}/featuretypes',
            auth=self.auth,
            headers={'Content-type': 'application/json'},
            data=json.dumps(body))

        if response.status_code == 201:
            return print('Adding new featuretype was successful')
        else:
            raise GeoserverAPIError(
                f'{response.status_code}:{response.content}')

    def __create_coveragestore(self):

        body = {
            'coverageStore': {
                'name':
                self.datastore,
                'description':
                '',
                'type':
                'GeoTIFF',
                'enabled':
                True,
                'workspace': {
                    'name':
                    self.workspace,
                    'href':
                    f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}.json'
                },
                '_default':
                False,
                'url':
                f'file:{self.file}',
                'coverages':
                f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/coveragestores/{self.datastore}/coverages.json'
            }
        }

        response = requests.post(
            f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/coveragestores',
            auth=self.auth,
            headers={'Content-type': 'application/json'},
            data=json.dumps(body))

        if response.status_code == 201:
            return print('Adding new coveragestore was successful')
        else:
            raise GeoserverAPIError(
                f'{response.status_code}:{response.content}')

    def __create_coverage(self):

        raster_file = self.file
        with rio.open(raster_file, 'r') as src:
            minx, miny, maxx, maxy = src.bounds
            crs_from = src.crs
            proj = pyproj.Transformer.from_crs(crs_from,
                                               pyproj.CRS.from_epsg(4326))
            miny_4326, minx_4326, maxy_4326, maxx_4326 = proj.transform_bounds(
                *src.bounds)

        body = {
            'coverage': {
                'name': self.name,
                'nativeName': self.name,
                'namespace': {
                    'name':
                    self.workspace,
                    'href':
                    f'https://{self.host}/geoserver/rest/namespaces/{self.workspace}.json'
                },
                'title': self.name,
                'description': 'Generated from GeoTIFF',
                'keywords': {
                    'string': [self.name, 'WCS', 'GeoTIFF']
                },
                'nativeCRS': {
                    '@class':
                    'projected',
                    '$':
                    'PROJCS["WGS 84 / UTM zone 19S", GEOGCS["WGS 84", DATUM["World Geodetic System 1984", SPHEROID["WGS 84", 6378137.0, '
                    '298.257223563, AUTHORITY["EPSG","7030"]], AUTHORITY["EPSG","6326"]], PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]], '
                    'UNIT["degree", 0.017453292519943295], AXIS["Geodetic longitude", EAST], AXIS["Geodetic latitude", NORTH], '
                    'AUTHORITY["EPSG","4326"]], PROJECTION["Transverse_Mercator", AUTHORITY["EPSG","9807"]], PARAMETER["central_meridian", '
                    '-69.0], PARAMETER["latitude_of_origin", 0.0], PARAMETER["scale_factor", 0.9996], PARAMETER["false_easting", 500000.0], '
                    'PARAMETER["false_northing", 10000000.0], UNIT["m", 1.0], AXIS["Easting", EAST], AXIS["Northing", NORTH], AUTHORITY["EPSG",'
                    '"32719"]]'
                },
                'srs': crs_from.to_string(),
                'nativeBoundingBox': {
                    'minx': minx,
                    'maxx': maxx,
                    'miny': miny,
                    'maxy': maxy,
                    'crs': {
                        '@class': 'projected',
                        '$': crs_from.to_string()
                    }
                },
                'latLonBoundingBox': {
                    'minx': minx_4326,
                    'maxx': maxx_4326,
                    'miny': miny_4326,
                    'maxy': maxy_4326,
                    'crs': 'EPSG:4326'
                },
                'projectionPolicy': 'REPROJECT_TO_DECLARED',
                'enabled': True,
                'metadata': {
                    'entry': {
                        '@key': 'dirName',
                        '$': self.datastore + '_' + self.name
                    }
                },
                'store': {
                    '@class':
                    'coverageStore',
                    'name':
                    f'{self.workspace}:{self.datastore}',
                    'href':
                    f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/coveragestores/{self.datastore}.json'
                },
                'serviceConfiguration': False,
                'simpleConversionEnabled': False,
                'nativeFormat': 'GeoTIFF',
                'grid': {
                    '@dimension': '2',
                    'range': {
                        'low': '0 0',
                        'high': '13038 8581'
                    },
                    'transform': {
                        'scaleX': 10,
                        'scaleY': -10,
                        'shearX': 0,
                        'shearY': 0,
                        'translateX': 343455,
                        'translateY': 8653645
                    },
                    'crs': crs_from.to_string()
                },
                'supportedFormats': {
                    'string': [
                        'ArcGrid', 'GEOTIFF', 'GeoPackage (mosaic)', 'GIF',
                        'PNG', 'JPEG', 'TIFF', 'ImageMosaic'
                    ]
                },
                'interpolationMethods': {
                    'string': ['nearest neighbor', 'bilinear', 'bicubic']
                },
                'defaultInterpolationMethod': 'nearest neighbor',
                'dimensions': {
                    'coverageDimension': [{
                        'name': 'RED_BAND',
                        'description':
                        'GridSampleDimension[-Infinity,Infinity]',
                        'range': {
                            'min': '-inf',
                            'max': 'inf'
                        },
                        'nullValues': {
                            "double": [0]
                        },
                        'unit': 'W.m-2.Sr-1',
                        'dimensionType': {
                            'name': 'UNSIGNED_8BITS'
                        }
                    }, {
                        'name': 'GREEN_BAND',
                        'description':
                        'GridSampleDimension[-Infinity,Infinity]',
                        'range': {
                            'min': '-inf',
                            'max': 'inf'
                        },
                        'nullValues': {
                            "double": [0]
                        },
                        'unit': 'W.m-2.Sr-1',
                        'dimensionType': {
                            'name': 'UNSIGNED_8BITS'
                        }
                    }, {
                        'name': 'BLUE_BAND',
                        'description':
                        'GridSampleDimension[-Infinity,Infinity]',
                        'range': {
                            'min': '-inf',
                            'max': 'inf'
                        },
                        'nullValues': {
                            'double': [0]
                        },
                        'unit': 'W.m-2.Sr-1',
                        'dimensionType': {
                            'name': 'UNSIGNED_8BITS'
                        }
                    }]
                },
                'requestSRS': {
                    'string': [crs_from.to_string()]
                },
                'responseSRS': {
                    'string': [crs_from.to_string()]
                },
                'parameters': {
                    'entry': [{
                        'string': 'InputTransparentColor',
                        'null': ''
                    }, {
                        'string': ['SUGGESTED_TILE_SIZE', '512,512']
                    }, {
                        'string': 'RescalePixels',
                        'boolean': True
                    }]
                },
                'nativeCoverageName': self.name
            }
        }

        response = requests.post(
            f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/coveragestores/{self.datastore}/coverages',
            auth=self.auth,
            headers={'Content-type': 'application/json'},
            data=json.dumps(body))

        if response.status_code == 201:
            return print('Adding new coverage was successful')
        else:
            raise GeoserverAPIError(
                f'{response.status_code}:{response.content}')

    def __create_workspace(self):

        body = {
            'workspace': {
                'name':
                self.workspace,
                'isolated':
                False,
                'dataStores':
                f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/datastores.json',
                'coverageStores':
                f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/coveragestores.json',
                'wmsStores':
                f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/wmsstores.json',
                'wmtsStores':
                f'https://{self.host}/geoserver/rest/workspaces/{self.workspace}/wmtsstores.json'
            }
        }

        response = requests.post(
            f'https://{self.host}/geoserver/rest/workspaces',
            auth=self.auth,
            headers={'Content-type': 'application/json'},
            data=json.dumps(body))

        if response.status_code == 201:
            return print('Adding new workspace was successful')
        else:
            raise GeoserverAPIError(
                f'{response.status_code}:{response.content}')

    def __extract_name_from_filename(self):
        """This function defines the names of the datastore and layer based on
        the filename.

        This has to be adjusted for individual naming conventions.
        """
        datastore_name = self.file.split('/')[-1][:-4]
        layer_name = self.file.split('/')[-1][:-4]
        return datastore_name, layer_name

    def add_all_files(self):
        """This function creates layers for each file in the directory.

        Only .shp and .tif files are accepted so far.
        """
        if self.layer_type == 'vector':
            for file in os.listdir(self.directory):
                if file.endswith('.shp'):
                    self.file = self.directory / file
                    self.datastore = self.__extract_name_from_filename()[0]
                    self.name = self.__extract_name_from_filename()[1]

                    self.__create_layer()

        elif self.layer_type == 'raster':
            for file in os.listdir(self.directory):
                if file.endswith('.tif'):
                    self.file = self.directory / file
                    self.datastore = self.__extract_name_from_filename()[0]
                    self.name = self.__extract_name_from_filename()[1]

                    self.__create_layer()

    def add_one_file(self, filename):
        """This function creates a layer for a single file in the directory.

        Only .shp files are accepted for vector data so far and .tif
        files for raster data.
        """
        if self.layer_type == 'vector':
            if filename.endswith('.shp'):
                self.file = self.directory / filename
                self.datastore = self.__extract_name_from_filename()[0]
                self.name = self.__extract_name_from_filename()[1]

                self.__create_layer()
            else:
                raise FileFormatError(
                    'Wrong file format: Only .shp files are accepted')

        elif self.layer_type == 'raster':
            if file.endswith('.tif'):
                self.file = self.directory / filename
                self.datastore = self.__extract_name_from_filename()[0]
                self.name = self.__extract_name_from_filename()[1]

                self.__create_layer()
            else:
                raise FileFormatError(
                    'Wrong file format: Only .tif files are accepted')


class GeoserverAPIError(Exception):
    pass


class FileFormatError(Exception):
    pass

import pandas as pd
import geopandas as gpd
from timezonefinder import TimezoneFinder
import pytz
import shapely


def update_timezone(df, type):

    df['date_time'] = pd.to_datetime(df['date_time'])
    df['date_time'] = df['date_time'].dt.tz_localize('US/Pacific')
    
    tf = TimezoneFinder()

    if type=='movement':
        df_check_tz = df[['start_latitude', 'start_longitude']].sample(n=100, random_state=42)
        df_check_tz = df_check_tz.rename(columns={'start_latitude':'latitude',
                                                  'start_longitude':'longitude'})
    else:
        df_check_tz = df[['latitude', 'longitude']].sample(n=100, random_state=42)
        
    df_check_tz['time_zone'] = df_check_tz.apply(lambda row: tf.timezone_at(lng=row['longitude'], lat=row['latitude']), axis=1)

    if len(df_check_tz['time_zone'].unique()) == 1:
        time_zone=df_check_tz['time_zone'].unique()[0]
        print(f'Time Zone: {time_zone}')
    elif len(df_check_tz[df_check_tz['time_zone'] == df_check_tz['time_zone'].mode()[0]]) >= 90:
        time_zone=df_check_tz['time_zone'].mode()[0]
        print(f'Dominant Time Zone: {time_zone}')
    else:
        time_zone='US/Pacific'
        print(f'Difficulties with identifying one Time Zone: {df_check_tz['time_zone'].unique()}. \n US/Pacific is used.')

    # Convert timestamps to Asia/Jakarta time
    df['date_time'] = df['date_time'].dt.tz_convert(time_zone)

    return df

def define_grid(df, grid_size = 2400):
    
    df=df.to_crs(3857)
    minx, miny, maxx, maxy = df.total_bounds

    
    # Create an empty list to store grid polygons
    grid_polygons = []
    
    cell_index = 1
    
    geometry_unique = df['geometry'].unique()
    
    for point in geometry_unique:
        # Calculate the coordinates of the four corners of the grid square
        xmin = point.x - (grid_size / 2)
        xmax = point.x + (grid_size / 2)
        ymin = point.y - (grid_size / 2)
        ymax = point.y + (grid_size / 2)
        
        # Define the corner coordinates of the grid square
        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        
        # Create a polygon representing the grid square
        grid_polygon = shapely.geometry.Polygon(corners)
        
        # Append the grid polygon to the list
        grid_polygons.append({'id': cell_index, 'geometry': grid_polygon})
    
        # Increment cell index
        cell_index += 1
    
    # Create a GeoDataFrame from the list of grid polygons
    gdf_grid = gpd.GeoDataFrame(grid_polygons, crs='3857')

    return gdf_grid
    
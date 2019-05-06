"""
Example of spatial graph

Use other attribute for coloring like edge weight

"""

def test_graph():
    import geoplotlib
    from geoplotlib.utils import read_csv

    movement_data = read_csv('./data/graph_movement.csv')

    geoplotlib.graph(movement_data,
                     src_lat='SourceLat',
                     src_lon='SourceLon',
                     dest_lat='TargetLat',
                     dest_lon='TargetLon',
                     alpha=80,
                     linewidth=2,
                     color='inferno',
                     color_by='Weight',levels=10)

    geoplotlib.show()


if __name__ == '__main__':

    test_graph()



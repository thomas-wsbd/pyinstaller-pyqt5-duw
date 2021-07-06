import folium, io, sys, json, pickle, os

import plotly.express as px
import plotly

from folium.plugins import Draw
from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInterceptor

# help functions
import branca.colormap as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import colors as colors
from rasterio import features


def getclasarray():
    array = np.load(os.path.join("data", "clas-array.npy"))
    array[array == 0] = np.nan
    return array


def getbounds():
    with open(os.path.join("data", "bounds.pickle"), "rb") as handle:
        bounds = pickle.load(handle)
    return bounds


def getcolormap():

    vmin = 1
    vmax = 7

    colormap = cm.linear.RdYlBu_11.scale(vmin, vmax)

    def reversed_colormap(existing):
        return cm.LinearColormap(
            colors=list(reversed(existing.colors)),
            vmin=existing.vmin,
            vmax=existing.vmax,
            caption="1: heel koude - tot 7: heel warme plek",
        )

    return reversed_colormap(colormap)


def mapvalue2color(value, cmap):
    """
    Map a pixel value of image to a color in the rgba format.
    As a special case, nans will be mapped totally transparent.

    Inputs
        -- value - pixel value of image, could be np.nan
        -- cmap - a linear colormap from branca.colormap.linear
    Output
        -- a color value in the rgba format (r, g, b, a)
    """
    if np.isnan(value):
        return (1, 0, 0, 0)
    else:
        return colors.to_rgba(cmap(value), 0.7)


def getpoly():
    from shapely.geometry import Polygon

    with open(os.path.join("data", "temp.pickle"), "rb") as infile:
        coords = pickle.load(infile)
    return Polygon(coords)


def plotframe():
    # based on last saved geom create plotframe
    array = np.load(os.path.join("data", "numpy-ndarray-l8-sd-25.npy"))
    shape = (array.shape[0], array.shape[1])  # shape row, cols
    with open(os.path.join("data", "affine.pickle"), "rb") as handle:
        affine = pickle.load(handle)
    with open(os.path.join("data", "index-25.pickle"), "rb") as handle:
        index = pickle.load(handle)  # contains date for each image
    poly = getpoly()
    mask = features.geometry_mask([poly], shape, affine, all_touched=True, invert=True)
    array[~mask] = float("nan")
    df = pd.DataFrame(
        np.nanmean(array, axis=(0, 1)),
        index=pd.to_datetime(list(index.values())),
        columns=["std-waarden"],
    )
    return df


def shapes():
    return [
        dict(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            y0=y0,
            x1=1,
            y1=y1,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0,
        )
        for y0, y1, color in zip(
            [5, 2, 1, 0.5, -0.5, -1, -2],
            [2, 1, 0.5, -0.5, -1, -2, -5],
            [
                "red",
                "orange",
                "gold",
                "lightyellow",
                "lightblue",
                "blue",
                "navy",
            ],
        )
    ]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.tr("DUW - Hittestressmonitoring APP"))
        self.setMinimumSize(1600, 1200)
        self.UI()
        self.Map()

    def UI(self):
        btn = QPushButton("Maak grafiek", self)
        btn.clicked.connect(self.onClick)
        btn.setFixedSize(120, 50)

        text = QLabel(
            "<B>DUW - Hittestressmonitor APP</B><BR>1. Teken een gebied in op de kaart [door op de vijfhoek te klikken of het vierkant]<BR>2. Klik op jouw ingetekende gebied [hierdoor sla je jouw ingetekende gebied op]<BR>3. Klik op OK<BR>4. Klik op Maak grafiek<BR>Bron; Landsat7 & 8"
        )

        self.view = QWebEngineView()
        self.view.setContentsMargins(20, 20, 20, 20)

        # central container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        lay = QHBoxLayout(central_widget)

        # menu container
        button_container = QWidget()
        vlay = QVBoxLayout(button_container)
        vlay.setSpacing(20)
        vlay.addStretch()
        vlay.addWidget(text)
        vlay.addWidget(btn)

        vlay.addStretch()
        lay.addWidget(button_container)
        lay.addWidget(self.view, stretch=1)

    def Map(self):
        array = getclasarray()
        colormap = getcolormap()
        bounds = getbounds()
        coordinate = [
            (bounds.bottom + bounds.top) / 2,
            (bounds.left + bounds.right) / 2,
        ]

        m = folium.Map(location=coordinate, zoom_start=13, tiles="CartoDB positron")

        folium.raster_layers.ImageOverlay(
            image=array,
            opacity=0.4,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            colormap=lambda value: mapvalue2color(value, colormap),
        ).add_to(m)
        m.add_child(colormap)

        # add draw component
        draw = Draw(
            draw_options={
                "polyline": False,
                "rectangle": True,
                "polygon": True,
                "circle": False,
                "marker": False,
                "circlemarker": False,
            },
            edit_options={"edit": False},
        )
        m.add_child(draw)

        # save map data to data object
        data = io.BytesIO()
        m.save(data, close_file=False)

        # set map to central widget, add listener to return drawn poly
        page = WebEnginePage(self.view)
        self.view.setPage(page)
        self.view.setHtml(data.getvalue().decode())

    def onClick(self):
        # click on button opens second window
        self.SW = SecondWindow()
        self.SW.resize(800, 600)
        self.SW.show()


class WebEnginePage(QWebEnginePage):
    # reads drawn shapes and saves them
    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        coords_dict = json.loads(msg)
        coords = coords_dict["geometry"]["coordinates"][
            0
        ]  # extract the coordinates from the message
        with open(os.path.join("..", "data", "temp.pickle"), "wb") as outfile:
            pickle.dump(coords, outfile)


class SecondWindow(QMainWindow):
    # if second window opens make the graph
    def __init__(self):
        super(SecondWindow, self).__init__()
        self.view = QWebEngineView()
        self.show_graph()
        self.setCentralWidget(self.view)

    def show_graph(self):
        df = plotframe()
        fig = px.scatter(
            df,
            color_discrete_sequence=["black"],
            trendline="ols",
            trendline_color_override="grey",
        ).update_traces(
            marker=dict(size=8),
        )
        fig.update_yaxes(range=[-5, 5], dtick=1).update_layout(
            title="oppervlaktetemperatuur over tijd voor laatst opgeslagen gebied",
            xaxis_title="datum",
            yaxis_title="std van gemiddelde",
            shapes=shapes(),
        )

        # set html to view
        self.view.setHtml(fig.to_html(include_plotlyjs="cdn"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MW = MainWindow()
    MW.resize(1200, 800)
    MW.show()
    sys.exit(app.exec_())

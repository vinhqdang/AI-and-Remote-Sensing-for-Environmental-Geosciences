"""Event metadata for the nine Pacific Rim earthquakes analysed."""
EVENTS = {
    # key: (label, USGS ComCat id, ScienceBase inventory item, facet (layer) name, mapping-extent facet or None)
    "kaikoura2016":   dict(label="Kaikōura 2016 (NZ)",        mw=7.8, comcat="us1000778i", sb=None, facet=None, extent=None),
    "northridge1994": dict(label="Northridge 1994 (USA)",     mw=6.7, comcat="ci3144585", sb="58595917e4b03639a6025f30", facet="Harp_and_Jibson_1995", extent=None),
    "niigata2004":    dict(label="Niigata-Chuetsu 2004 (JP)", mw=6.6, comcat="usp000d6vk", sb="5874bd79e4b0a829a320bbbe", facet="Sekiguchi_and_Sato_2006", extent="mapping_extent"),
    "iwate2008":      dict(label="Iwate-Miyagi 2008 (JP)",    mw=6.9, comcat="usp000g9h6", sb="5b59ec98e4b0610d7f4da850", facet="Yagi_and_others_2009", extent=None),
    "aisen2007":      dict(label="Aisén 2007 (CL)",           mw=6.2, comcat="usp000fa97", sb="586fe5a7e4b01a71ba0c5f06", facet="Gorum_and_others_2014", extent=None),
    "denali2002":     dict(label="Denali 2002 (USA)",         mw=7.9, comcat="ak002e435qpj", sb="5874a336e4b0a829a320bb15", facet="Gorum_and_others_2014", extent=None),
    "cinchona2009":   dict(label="Cinchona 2009 (CR)",        mw=6.1, comcat="usp000gscg", sb="61ead71bd34e8b818ad9f44b", facet="Ruiz_2019", extent=None),
    "porgera2018":    dict(label="Porgera 2018 (PNG)",        mw=7.5, comcat="us2000d7q6", sb="61f04680d34e8b818adc32a8", facet="Tanyas_and_others_2020", extent="mapping_extent"),
    "palu2018":       dict(label="Palu 2018 (ID)",            mw=7.5, comcat="us1000h3p4", sb="61f196a1d34e8b818add4fdf", facet="Zhao_2021_pts", extent="mapping_extent"),
}

import synalinks
from typing import Literal, Union


class City(synalinks.Entity):
    label: Literal["City"]
    name: str = synalinks.Field(
        description="The name of a city, such as 'Paris' or 'New York'.",
    )


class Country(synalinks.Entity):
    label: Literal["Country"]
    name: str = synalinks.Field(
        description="The name of a country, such as 'France' or 'Canada'.",
    )


class Place(synalinks.Entity):
    label: Literal["Place"]
    name: str = synalinks.Field(
        description="The name of a specific place, which could be a landmark, building, or any point of interest, such as 'Eiffel Tower' or 'Statue of Liberty'.",
    )


class Event(synalinks.Entity):
    label: Literal["Event"]
    name: str = synalinks.Field(
        description="The name of an event, such as 'Summer Olympics 2024' or 'Woodstock 1969'.",
    )


class IsCapitalOf(synalinks.Relation):
    subj: City = synalinks.Field(
        description="The city entity that serves as the capital.",
    )
    label: Literal["IsCapitalOf"]
    obj: Country = synalinks.Field(
        description="The country entity for which the city is the capital.",
    )


class IsCityOf(synalinks.Relation):
    subj: City = synalinks.Field(
        description="The city entity that is a constituent part of a country.",
    )
    label: Literal["IsCityOf"]
    obj: Country = synalinks.Field(
        description="The country entity that the city is part of.",
    )


class IsLocatedIn(synalinks.Relation):
    subj: Union[Place] = synalinks.Field(
        description="The place entity that is situated within a larger geographical area.",
    )
    label: Literal["IsLocatedIn"]
    obj: Union[City, Country] = synalinks.Field(
        description="The city or country entity where the place is geographically located.",
    )


class TookPlaceIn(synalinks.Relation):
    subj: Event = synalinks.Field(
        description="The event entity that occurred in a specific location.",
    )
    label: Literal["TookPlaceIn"]
    obj: Union[City, Country] = synalinks.Field(
        description="The city or country entity where the event occurred.",
    )

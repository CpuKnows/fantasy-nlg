# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader.processors import MapCompose, TakeFirst


class PlayerStats(scrapy.Item):
    url = scrapy.Field(output_processor=TakeFirst())
    player = scrapy.Field(output_processor=TakeFirst())
    position = scrapy.Field(output_processor=TakeFirst())
    team = scrapy.Field(output_processor=TakeFirst())
    passing = scrapy.Field()
    rushing = scrapy.Field()
    receiving = scrapy.Field()

# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader.processors import MapCompose, TakeFirst


def strip_spaces(text):
    return text.strip()


def strip_position(text):
    return text.strip('\r\n -')


def strip_header_name(text):
    idx = text.index('|')
    return text[:idx].strip()


def strip_header_position(text):
    idx = text.index('|')
    text = text[idx + 1:]
    idx = text.index('|')
    return text[:idx].strip()


def header_map_position(text):
    if text == 'Quarterback':
        return 'QB'
    elif text == 'Tight End':
        return 'TE'
    elif text == 'Running Back':
        return 'RB'
    elif text == 'Wide Receiver':
        return 'WR'
    else:
        return text


def remove_nbsp(text):
    return text.replace('\xa0', ' ')


def to_int(text):
    return int(text)


def to_float(text):
    return float(text)


class PlayerLink(scrapy.Item):
    player_name = scrapy.Field(output_processor=TakeFirst())
    player_link = scrapy.Field(output_processor=TakeFirst())


class PlayerNews(scrapy.Item):
    url = scrapy.Field(output_processor=TakeFirst())
    player = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())
    position = scrapy.Field(input_processor=MapCompose(strip_position), output_processor=TakeFirst())
    team = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())
    report = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())
    impact = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())
    source_link = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())
    source_text = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())
    date = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())


class PlayerStats(scrapy.Item):
    url = scrapy.Field(output_processor=TakeFirst())
    player = scrapy.Field(input_processor=MapCompose(strip_header_name), output_processor=TakeFirst())
    position = scrapy.Field(input_processor=MapCompose(strip_header_position, header_map_position),
                            output_processor=TakeFirst())
    team = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())
    week = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    date = scrapy.Field(input_processor=MapCompose(strip_spaces, remove_nbsp), output_processor=TakeFirst())
    opp = scrapy.Field(input_processor=MapCompose(strip_spaces), output_processor=TakeFirst())
    reception = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    rec_yards = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    rec_avg = scrapy.Field(input_processor=MapCompose(strip_spaces, to_float), output_processor=TakeFirst())
    rec_td = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    rush_attempts = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    rush_yards = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    rush_avg = scrapy.Field(input_processor=MapCompose(strip_spaces, to_float), output_processor=TakeFirst())
    rush_td = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    pass_completions = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    pass_attempts = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    pass_percent = scrapy.Field(input_processor=MapCompose(strip_spaces, to_float), output_processor=TakeFirst())
    pass_yards = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    pass_ya = scrapy.Field(input_processor=MapCompose(strip_spaces, to_float), output_processor=TakeFirst())
    pass_td = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    pass_int = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    fumb_lost = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    ko_ret_yards = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    ko_ret_td = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    punt_ret_yards = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())
    punt_ret_td = scrapy.Field(input_processor=MapCompose(strip_spaces, to_int), output_processor=TakeFirst())

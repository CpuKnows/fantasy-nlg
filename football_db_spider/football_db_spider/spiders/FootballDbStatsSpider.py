import json
import string
import scrapy
from scrapy.loader import ItemLoader

from football_db_spider.items import PlayerStats


class FootballDbStatsSpider(scrapy.Spider):
    name = "football_db_stats_spider"

    def __init__(self, *args, **kwargs):
        super(FootballDbStatsSpider, self).__init__(*args, **kwargs)

    def start_requests(self):
        for letter in list(string.ascii_uppercase):
            yield scrapy.Request(url='https://www.footballdb.com/players/current.html?letter=' + letter,
                                 callback=self.parse)

    def parse(self, response):
        for row in response.xpath('//div[@class="divtable divtable-striped"]/div[@class="tr"]'):
            if row.xpath('./div[2]/text()').extract_first() in ['QB', 'RB', 'WR', 'TE']:
                yield scrapy.Request(url='https://www.footballdb.com' + row.xpath('./div[1]/a/@href').extract_first() +
                                         '/gamelogs',
                                     callback=self.parse_player)

    def parse_player(self, response):
        passing_headers = ['Date', 'Opp', 'Att', 'Cmp', 'Pct', 'Yds', 'YPA', 'TD', 'Int', 'Lg', 'Sack', 'Rate',
                           'Result']
        rushing_headers = ['Date', 'Opp', 'Att', 'Yds', 'Avg', 'Lg', 'TD', 'FD', 'Result']
        receiving_headers = ['Date', 'Opp', 'Rec', 'Yds', 'Avg', 'Lg', 'TD', 'FD', 'Tar', 'YAC', 'Result']
        passing_data = []
        rushing_data = []
        receiving_data = []

        stats_table_sel = response.xpath('//table[@class="statistics scrollable"]')
        for stats_table in stats_table_sel:
            header_list = list(stats_table.xpath('.//thead/tr/th/text()').extract())
            if set(header_list) == set(passing_headers):
                for row in stats_table.xpath('./tbody/tr'):
                    passing_data.append(row.xpath('./td/text()').extract() + row.xpath('./td/a/text()').extract())
            elif set(header_list) == set(rushing_headers):
                for row in stats_table.xpath('./tbody/tr'):
                    rushing_data.append(row.xpath('./td/text()').extract() + row.xpath('./td/a/text()').extract())
            elif set(header_list) == set(receiving_headers):
                for row in stats_table.xpath('./tbody/tr'):
                    receiving_data.append(row.xpath('./td/text()').extract() + row.xpath('./td/a/text()').extract())

        loader = ItemLoader(PlayerStats(), response)
        loader.add_value('url', response.url)
        loader.add_value('player', response.xpath('//div[@class="teamlabel"]/text()').extract_first())
        for i in list(response.xpath('//div[@id="playerbanner"]/text()').extract()):
            if i.strip() in ['QB', 'RB', 'WR', 'TE']:
                loader.add_value('position', i.strip())
                break
        loader.add_value('team', response.xpath('//div[@id="playerbanner"]/b/a/text()').extract_first())
        loader.add_value('passing', passing_data)
        loader.add_value('rushing', rushing_data)
        loader.add_value('receiving', receiving_data)
        yield loader.load_item()

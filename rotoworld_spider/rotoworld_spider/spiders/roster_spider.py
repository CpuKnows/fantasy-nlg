import scrapy
from scrapy.loader import ItemLoader

from rotoworld_spider.items import PlayerLink


class RosterSpider(scrapy.Spider):
    name = "roster_spider"

    def start_requests(self):
        urls = [
            'http://www.rotoworld.com/teams/depth-charts/nfl.aspx'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for division in response.xpath('//select[@name="ctl00$cp1$ddlDivisions"]//option/@value').extract()[::-1]:
            yield scrapy.FormRequest.from_response(
                response,
                formid='ctl01',
                formdata={
                    '__EVENTTARGET': 'ctl00$cp1$ddlDivisions',
                    '__EVENTARGUMENT': '',
                    'ctl00$cp1$ddlDivisions': division
                },
                dont_click=True,
                callback=self.parse_roster
            )

    def parse_roster(self, response):
        for team in response.xpath('//table[starts-with(@id, "cp1_tblTeam")]'):
            for tr in team.xpath('./tr/td/a'):
                loader = ItemLoader(PlayerLink(), tr)
                loader.add_xpath('player_name', './text()')
                loader.add_xpath('player_link', './@href')
                yield loader.load_item()

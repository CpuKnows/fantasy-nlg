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
        yield scrapy.Request(url=response.url, callback=self.parse_roster, dont_filter=True)
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
                dont_filter=True,
                callback=self.parse_roster
            )

    def parse_roster(self, response):
        position = ''
        for team in response.xpath('//table[starts-with(@id, "cp1_tblTeam")]'):
            for tr in team.xpath('./tr'):
                if tr.xpath('./@class').extract_first() == 'highlight-row':
                    position = tr.xpath('./td/b/text()').extract_first()

                loader = ItemLoader(PlayerLink(), tr)
                loader.add_xpath('player_name', './td[2]/a/text()')
                loader.add_xpath('player_link', './td[2]/a/@href')
                loader.add_value('player_position', position)
                yield loader.load_item()

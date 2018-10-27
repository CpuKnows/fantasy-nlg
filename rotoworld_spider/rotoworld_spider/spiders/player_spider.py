import json
import scrapy
from scrapy.loader import ItemLoader

from rotoworld_spider.items import PlayerNews


class PlayerSpider(scrapy.Spider):
    name = "player_spider"

    def __init__(self, player_links_file=None, *args, **kwargs):
        super(PlayerSpider, self).__init__(*args, **kwargs)
        self.player_links_file = player_links_file

    def start_requests(self):
        # Example url: 'http://www.rotoworld.com/recent/nfl/4186/marshawn-lynch'
        base_url = 'http://www.rotoworld.com/recent'
        urls = []
        with open(self.player_links_file, 'r') as f:
            for line in f:
                urls.append(base_url + json.loads(line)['player_link'][7:])

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        top_story = response.xpath('//div[@class="pp"]')
        loader = ItemLoader(PlayerNews(), top_story)
        loader.add_value('url', response.url)
        loader.add_xpath('player', './div[@class="playerdetails"]/div[@class="playername"]/h1/text()')
        loader.add_xpath('position', './div[@class="playerdetails"]/div[@class="playername"]/h1/text()')
        loader.add_xpath('team', './div[@class="playerdetails"]/div[@class="playername"]/' +
                         'table[@id="cp1_ctl00_tblPlayerDetails"]/tr/td[2]/a/text()')

        blurb = loader.nested_xpath('./div[@class="playernews"]')
        blurb.add_xpath('report', './div[@class="report"]/text()')
        blurb.add_xpath('impact', './div[@class="impact"]/text()')
        blurb.add_xpath('source_link', './div[@class="info"]/div[@class="source"]/a/@href')
        blurb.add_xpath('source_text', './div[@class="info"]/div[@class="source"]/a/text()')
        blurb.add_xpath('date', './div[@class="impact"]/span[@class="date"]/text()')

        yield loader.load_item()

        for news in response.xpath('//div[@class="pb"]'):
            loader = ItemLoader(PlayerNews(), news)
            loader.add_value('url', response.url)

            headline = loader.nested_xpath('./div[@class="headline"]/div[@class="player"]')
            headline.add_xpath('player', './a[1]/text()')
            headline.add_xpath('position', './text()')
            headline.add_xpath('team', './a[2]/text()')

            blurb = loader.nested_xpath('./div[starts-with(@id, "cp1_ctrlPlayerNews_rptBlurbs_floatingcontainer_")]')
            blurb.add_xpath('report', './div[@class="report"]/p/text()')
            blurb.add_xpath('impact', './div[@class="impact"]/text()')
            blurb.add_xpath('source_link', './div[@class="info"]/div[@class="source"]/a/@href')
            blurb.add_xpath('source_text', './div[@class="info"]/div[@class="source"]/a/text()')
            blurb.add_xpath('date', './div[@class="info"]/div[@class="date"]/text()')

            yield loader.load_item()

import json
import scrapy
from scrapy.loader import ItemLoader

from rotoworld_spider.items import PlayerStats


class StatsSpider(scrapy.Spider):
    name = "stats_spider"

    def __init__(self, player_links_file=None, *args, **kwargs):
        super(StatsSpider, self).__init__(*args, **kwargs)
        self.player_links_file = player_links_file
        self.stats_cols = ('week', 'date', 'opp', 'reception', 'rec_yards', 'rec_avg', 'rec_td', 'rush_attempts',
                           'rush_yards', 'rush_avg', 'rush_td', 'pass_completions', 'pass_attempts', 'pass_percent',
                           'pass_yards', 'pass_ya', 'pass_td', 'pass_int', 'fumb_lost', 'ko_ret_yards', 'ko_ret_td',
                           'punt_ret_yards', 'punt_ret_td')

    def start_requests(self):
        # Example url: 'http://www.rotoworld.com/log/nfl/4186/marshawn-lynch'
        base_url = 'http://www.rotoworld.com/log'
        urls = []
        with open(self.player_links_file, 'r') as f:
            for line in f:
                urls.append(base_url + json.loads(line)['player_link'][7:])

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        stats_table_sel = response.xpath('//div[@id="cp1_pnlStatControls"]/table')[0]
        headers = []
        for header in stats_table_sel.xpath('./tr[2]//th'):
            headers.append((header.xpath('./text()').extract_first(),
                            int(header.xpath('./@colspan').extract_first())))

        col_list = []
        for header in headers:
            if header[0] == 'Game' and header[1] == 3:
                col_list.extend(['week', 'date', 'opp'])
            elif header[0] == 'Receiving' and header[1] == 4:
                col_list.extend(['reception', 'rec_yards', 'rec_avg', 'rec_td'])
            elif header[0] == 'Rushing' and header[1] == 4:
                col_list.extend(['rush_attempts', 'rush_yards', 'rush_avg', 'rush_td'])
            elif header[0] == 'Passing' and header[1] == 7:
                col_list.extend(['pass_completions', 'pass_attempts', 'pass_percent', 'pass_yards', 'pass_ya',
                                 'pass_td', 'pass_int'])
            elif header[0] == 'Fumb.' and header[1] == 1:
                col_list.extend(['fumb_lost'])
            elif header[0] == 'KO Ret' and header[1] == 2:
                col_list.extend(['ko_ret_yards', 'ko_ret_td'])
            elif header[0] == 'Punt Ret' and header[1] == 2:
                col_list.extend(['punt_ret_yards', 'punt_ret_td'])

        for idx, tr in enumerate(stats_table_sel.xpath('.//tr')):
            if idx < 3:
                pass
            else:
                if 'Game scheduled' in tr.xpath('./td[4]/text()').extract_first():
                    pass
                else:
                    loader = ItemLoader(PlayerStats(), tr)
                    loader.add_value('url', response.url)
                    loader.add_value('player', response.xpath('//div[@class="playername"]/h1/text()').extract_first())
                    loader.add_value('position', response.xpath('//div[@class="playername"]/h1/text()').extract_first())
                    loader.add_value('team', response.xpath('//table[@id="cp1_ctl00_tblPlayerDetails"]' +
                                                            '/tr/td[2]/a/text()').extract_first())

                    cols = tr.xpath('.//td/text()').extract()
                    for k, v in zip(col_list, cols):
                        loader.add_value(k, v)
                    yield loader.load_item()

from contexeval.prep_wcep import parse_day, prep, story_title, MIN_EVENTS

def _art(id, origin="CommonCrawl", pub="Aug 23, 2016", title="T", text="body"):
    return {"id": id, "origin": origin, "publish_data": pub, "title": title, "text": text}

def _cluster(id, date, links, arts, summary="s"):
    return {"id": id, "date": date, "wiki_links": links, "articles": arts, "summary": summary}

def test_parse_day_formats_and_garbage():
    assert parse_day("Aug 23, 2016") == "2016-08-23"
    assert parse_day("2016-08-25") == "2016-08-25"
    assert parse_day(None) is None
    assert parse_day("None") is None
    assert parse_day("sometime") is None

def test_story_title_unquotes():
    assert story_title("/wiki/Rodrigo_Duterte") == "Rodrigo Duterte"
    assert story_title("/wiki/Caf%C3%A9_Society") == "Café Society"

def _many(link, n, start=0):
    # dated 2017-01-01 so they don't share a (story, day) gold key with the clusters under test
    return [_cluster(start + i, "2017-01-01", [link], [_art(f"x{start+i}", pub=None)]) for i in range(n)]

def test_gold_uses_editor_cited_only_and_skips_rare_entities():
    target = _cluster(999, "2017-01-10", ["/wiki/Brexit", "/wiki/Rare"], [
        _art("cited", origin="WCEP", pub="Jan 09, 2017"),
        _art("auto", origin="CommonCrawl", pub="Jan 10, 2017"),
    ], summary="Parliament votes.")
    clusters = _many("/wiki/Brexit", MIN_EVENTS - 1) + [target]
    articles, stories, gold = prep(clusters)
    assert [s["story"] for s in stories] == ["/wiki/Brexit"]
    assert stories[0]["need"] == "news about Brexit"
    g = next(g for g in gold if g["events"] == [999])
    assert g["story"] == "/wiki/Brexit" and g["day"] == "2017-01-10"
    assert g["articles"] == ["cited"]
    assert g["summaries"] == ["Parliament votes."]

def test_article_fields_fallbacks_and_dedup():
    clusters = _many("/wiki/Brexit", MIN_EVENTS) + [
        _cluster(500, "2017-02-01", ["/wiki/Brexit"], [
            _art("a", pub=None, title=None, text=None),
            _art("b", pub="Jan 01, 1999"),          # out of range -> event date
            _art("c", pub="Feb 03, 2017", text="z" * 5000),
        ]),
        _cluster(501, "2017-03-01", ["/wiki/Brexit"], [_art("a", pub="Mar 01, 2017")]),  # dup id
    ]
    by = {a["id"]: a for a in prep(clusters)[0]}
    assert by["a"]["day"] == "2017-02-01" and by["a"]["text"] == "" and by["a"]["event"] == 500
    assert by["b"]["day"] == "2017-02-01"
    assert by["c"]["day"] == "2017-02-03" and len(by["c"]["text"]) == 1000
    days = [a["day"] for a in prep(clusters)[0]]
    assert days == sorted(days)

def test_text_is_title_plus_one_body_paragraph():
    # Contex's text parser makes one node per blank-line paragraph; one body paragraph keeps
    # each article at two nodes (title, body) instead of ~6.
    clusters = _many("/wiki/Brexit", MIN_EVENTS) + [_cluster(600, "2017-02-01", ["/wiki/Brexit"], [
        _art("p", title="Headline", text="First para.\n\nSecond   para.\nThird.")])]
    by = {a["id"]: a for a in prep(clusters)[0]}
    assert by["p"]["text"] == "Headline\n\nFirst para. Second para. Third."

def test_day_from_time_field_within_a_week_of_the_event_else_event_date():
    # val/test articles and every editor-cited article carry only "time"; far-off times are
    # archive/crawl times, not publish dates.
    def t(id, time):
        return {"id": id, "origin": "WCEP", "time": time, "title": "T", "text": "b"}
    clusters = _many("/wiki/Brexit", MIN_EVENTS) + [_cluster(700, "2019-01-06", ["/wiki/Brexit"], [
        t("near", "2019-01-07 14:55:49+00:00"),
        t("crawl", "2020-10-06 11:40:29.404639+00:00"),
        t("none", "None"),
    ])]
    by = {a["id"]: a for a in prep(clusters)[0]}
    assert by["near"]["day"] == "2019-01-07"
    assert by["crawl"]["day"] == "2019-01-06" and by["none"]["day"] == "2019-01-06"

def test_publish_data_far_from_event_falls_back_to_event_date():
    clusters = _many("/wiki/Brexit", MIN_EVENTS) + [_cluster(701, "2017-02-01", ["/wiki/Brexit"], [
        _art("far", pub="Mar 20, 2017")])]
    assert {a["id"]: a for a in prep(clusters)[0]}["far"]["day"] == "2017-02-01"

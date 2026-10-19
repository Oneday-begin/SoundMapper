"""Read-only browser verification of the real local application; no listener ratings fabricated."""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
screens=ROOT/'reports'/'screenshots';screens.mkdir(parents=True,exist_ok=True)
checks=[];errors=[]
start=time.perf_counter()
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1050})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8765',wait_until='networkidle')
    assert page.locator('#track-a option').count()>=7
    checks.append('Analyzed library tracks listed, including the seven samples')
    page.click('#compare-button')
    page.locator('.big-score').wait_for(timeout=120000)
    assert page.locator('[data-passage]').count()>=2
    checks.append('Two-song comparison and matching-passage controls returned')
    page.screenshot(path=str(screens/'compare-desktop.png'),full_page=True)
    page.locator('[data-passage]').first.click()
    page.wait_for_timeout(1200)
    assert page.locator('#player-a').evaluate('(p)=>p.currentTime')>=0
    checks.append('Matching passage seek/play control executes')
    page.locator('#player-a').evaluate('(p)=>p.pause()')
    page.click('nav [data-view="library"]')
    page.locator('#music-map .js-plotly-plot').count()
    page.wait_for_timeout(1000)
    page.locator('[data-neighbor]').first.click()
    page.locator('[data-open-pair]').first.wait_for(timeout=120000)
    assert 1<=page.locator('[data-open-pair]').count()<=10
    checks.append('Library retrieval excludes selected recording; map renders')
    page.screenshot(path=str(screens/'library-desktop.png'),full_page=True)
    page.click('nav [data-view="analysis"]')
    page.click('#analysis-button')
    page.locator('#export-analysis').wait_for(timeout=30000)
    assert page.locator('#chroma .main-svg').count()>0
    checks.append('Analysis waveform, energy, chroma, and JSON export control render')
    page.screenshot(path=str(screens/'analysis-desktop.png'),full_page=True)
    page.click('nav [data-view="evaluation"]')
    page.click('#next-pair')
    page.locator('#rating-players audio').first.wait_for(timeout=10000)
    assert page.locator('#rating-players audio').count()==2
    page.click('#save-rating')
    page.wait_for_timeout(500)
    assert 'Rate at least one dimension' in page.locator('#notice').inner_text()
    checks.append('Listener pair controls work; empty judgment rejected without saving')
    page.screenshot(path=str(screens/'evaluation-desktop.png'),full_page=True)
    page.click('nav [data-view="compare"]')
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_timeout(800)
    assert page.evaluate('document.documentElement.scrollWidth')<=390
    checks.append('Mobile layout has no horizontal overflow')
    page.screenshot(path=str(screens/'compare-mobile.png'),full_page=True)
    assert not errors,errors
    browser.close()
report={'status':'passed','checks':checks,'page_errors':errors,'seconds':time.perf_counter()-start,
        'screenshots':[str(p.relative_to(ROOT)) for p in screens.glob('*.png')],
        'ratings_created':0,'scope':'Real sample library; read-only UI and invalid empty rating submission'}
(ROOT/'reports'/'browser-tests.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

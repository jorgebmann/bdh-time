#!/usr/bin/env python3
"""
Filter news articles to keep only those relevant to a specific ticker.

Filters out articles where the ticker is mentioned incidentally but isn't the focus.
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter

# Ticker keyword database (case-insensitive keywords)
# Maps ticker symbols to sets of relevant keywords
TICKER_KEYWORDS = {
    'AAPL.US': {
        'apple', 'aapl', 'iphone', 'ipad', 'macbook', 'mac', 'ios', 'app store',
        'tim cook', 'cook', 'cupertino', 'apple watch', 'airpods', 'apple tv',
        'apple pay', 'icloud', 'siri', 'apple silicon', 'm-series', 'm1', 'm2', 'm3',
        'apple music', 'apple services', 'apple store'
    },
    'MSFT.US': {
        'microsoft', 'msft', 'azure', 'windows', 'office', 'xbox', 'surface',
        'satya nadella', 'teams', 'linkedin', 'github', 'visual studio', 'dotnet',
        'power bi', 'dynamics', 'outlook', 'onedrive', 'bing', 'skype'
    },
    'AMZN.US': {
        'amazon', 'amzn', 'aws', 'prime', 'alexa', 'echo', 'kindle', 'fire tv',
        'andy jassy', 'amazon web services', 's3', 'ec2', 'lambda', 'prime video',
        'amazon music', 'twitch', 'whole foods', 'amazon fresh'
    },
    'NVDA.US': {
        'nvidia', 'nvda', 'gpu', 'cuda', 'tensorrt', 'rtx', 'gtx', 'jensen huang',
        'ai chips', 'data center', 'gaming', 'autonomous vehicles', 'omniverse',
        'dlss', 'ray tracing', 'geforce', 'quadro', 'tesla', 'a100', 'h100'
    },
    'GOOGL.US': {
        'google', 'googl', 'alphabet', 'sundar pichai', 'search', 'youtube',
        'android', 'chrome', 'gmail', 'maps', 'cloud', 'pixel', 'nest', 'waymo',
        'deepmind', 'tensorflow', 'adwords', 'adsense', 'play store'
    },
    'GOOG.US': {
        'google', 'goog', 'alphabet', 'sundar pichai', 'search', 'youtube',
        'android', 'chrome', 'gmail', 'maps', 'cloud', 'pixel', 'nest', 'waymo',
        'deepmind', 'tensorflow', 'adwords', 'adsense', 'play store'
    },
    'META.US': {
        'meta', 'facebook', 'fb', 'instagram', 'whatsapp', 'oculus', 'quest',
        'mark zuckerberg', 'reels', 'metaverse', 'vr', 'ar', 'horizon', 'threads'
    },
    'TSLA.US': {
        'tesla', 'tsla', 'elon musk', 'model s', 'model 3', 'model x', 'model y',
        'cybertruck', 'supercharger', 'autopilot', 'fsd', 'gigafactory', 'battery',
        'electric vehicle', 'ev', 'solar', 'powerwall', 'megapack'
    },
    'AVGO.US': {
        'broadcom', 'avgo', 'semiconductor', 'chips', 'networking', 'wifi',
        'bluetooth', 'ethernet', 'data center', '5g', 'infrastructure'
    },
    'COST.US': {
        'costco', 'cost', 'wholesale', 'warehouse', 'membership', 'kirkland',
        'bulk', 'retail', 'sam\'s club competitor'
    },
    'NFLX.US': {
        'netflix', 'nflx', 'streaming', 'original content', 'ted sarandos',
        'stranger things', 'squid game', 'subscription', 'netflix originals'
    },
    'AMD.US': {
        'amd', 'advanced micro devices', 'ryzen', 'epyc', 'radeon', 'lisa su',
        'cpu', 'gpu', 'data center', 'gaming', 'zen', 'rdna'
    },
    'ADBE.US': {
        'adobe', 'adbe', 'photoshop', 'illustrator', 'premiere', 'creative cloud',
        'pdf', 'acrobat', 'indesign', 'after effects', 'figma', 'shantanu narayen'
    },
    'CSCO.US': {
        'cisco', 'csco', 'networking', 'routers', 'switches', 'security',
        'webex', 'meraki', 'collaboration', 'chuck robbins'
    },
    'CMCSA.US': {
        'comcast', 'cmcsa', 'xfinity', 'nbcuniversal', 'peacock', 'broadband',
        'cable', 'internet', 'brian roberts'
    },
    'INTC.US': {
        'intel', 'intc', 'pat gelsinger', 'cpu', 'processors', 'xeon', 'core',
        'foundry', 'semiconductor', 'chips', 'data center'
    },
    'TXN.US': {
        'texas instruments', 'txn', 'ti', 'analog', 'embedded', 'semiconductor',
        'calculators', 'chips', 'electronics'
    },
    'AMGN.US': {
        'amgen', 'amgn', 'biotechnology', 'pharmaceuticals', 'drugs', 'oncology',
        'rheumatology', 'neuroscience', 'biosimilars'
    },
    'QCOM.US': {
        'qualcomm', 'qcom', 'snapdragon', '5g', 'modems', 'wireless', 'cristiano amon',
        'mobile chips', 'smartphone', 'connectivity'
    },
    'INTU.US': {
        'intuit', 'intu', 'quickbooks', 'turbotax', 'mint', 'credit karma',
        'small business', 'accounting', 'tax software', 'sasan goodarzi'
    },
    'ISRG.US': {
        'intuitive surgical', 'isrg', 'da vinci', 'robotic surgery', 'surgical robots',
        'minimally invasive', 'healthcare technology'
    },
    'VRTX.US': {
        'vertex pharmaceuticals', 'vrtx', 'cystic fibrosis', 'biotechnology',
        'drug development', 'rare diseases', 'pharmaceuticals'
    },
    'BKNG.US': {
        'booking', 'bkng', 'booking.com', 'priceline', 'kayak', 'agoda',
        'travel', 'hotels', 'reservations', 'glenn fogel'
    },
    'ADP.US': {
        'adp', 'automatic data processing', 'payroll', 'hr', 'human resources',
        'workforce management', 'benefits administration'
    },
    'REGN.US': {
        'regeneron', 'regn', 'biotechnology', 'pharmaceuticals', 'eylea', 'dupixent',
        'drug development', 'leonard schleifer'
    },
    'CDNS.US': {
        'cadence', 'cdns', 'eda', 'electronic design', 'semiconductor design',
        'chip design', 'verification', 'lip-bu tan'
    },
    'SNPS.US': {
        'synopsys', 'snps', 'eda', 'electronic design', 'semiconductor design',
        'chip design', 'verification', 'aart de geus'
    },
    'CRWD.US': {
        'crowdstrike', 'crwd', 'cybersecurity', 'endpoint protection', 'falcon',
        'threat intelligence', 'george kurtz', 'cloud security'
    },
    'MRVL.US': {
        'marvell', 'mrvl', 'semiconductor', 'networking', 'storage', 'data center',
        'chips', 'connectivity'
    },
    'KLAC.US': {
        'kla', 'klac', 'semiconductor equipment', 'wafer inspection', 'process control',
        'chip manufacturing', 'metrology'
    },
    'NXPI.US': {
        'nxp', 'nxpi', 'semiconductors', 'automotive', 'iot', 'connectivity',
        'nfc', 'secure elements', 'chips'
    },
    'CDW.US': {
        'cdw', 'it solutions', 'technology products', 'business technology',
        'hardware', 'software', 'services'
    },
    'FTNT.US': {
        'fortinet', 'ftnt', 'cybersecurity', 'firewall', 'network security',
        'ken xie', 'secure sd-wan', 'utm'
    },
    'ODFL.US': {
        'old dominion', 'odfl', 'freight', 'trucking', 'logistics', 'shipping',
        'ltl', 'less-than-truckload'
    },
    'CTAS.US': {
        'cintas', 'ctas', 'uniforms', 'business services', 'facility services',
        'safety', 'first aid', 'fire protection'
    },
    'ANSS.US': {
        'ansys', 'anss', 'simulation', 'engineering software', 'cae', 'fda',
        'finite element', 'computational fluid dynamics'
    },
    'TEAM.US': {
        'atlassian', 'team', 'jira', 'confluence', 'bitbucket', 'trello',
        'software development', 'project management', 'mike cannon-brookes'
    },
    'FAST.US': {
        'fastenal', 'fast', 'industrial supplies', 'fasteners', 'mro',
        'maintenance repair operations', 'distribution'
    },
    'PCAR.US': {
        'paccar', 'pcar', 'trucks', 'kenworth', 'peterbilt', 'daf',
        'heavy-duty trucks', 'commercial vehicles'
    },
    'EXPD.US': {
        'expeditors', 'expd', 'logistics', 'freight forwarding', 'customs',
        'supply chain', 'international shipping'
    },
    'IDXX.US': {
        'idexx', 'idxx', 'veterinary', 'diagnostics', 'laboratory', 'pet healthcare',
        'livestock', 'water testing'
    },
    'DXCM.US': {
        'dexcom', 'dxcm', 'diabetes', 'cgm', 'continuous glucose monitoring',
        'medical devices', 'healthcare'
    },
    'ZS.US': {
        'zscaler', 'zs', 'cloud security', 'zero trust', 'secure access',
        'jay chaudhry', 'network security', 'sase'
    },
    'BKR.US': {
        'baker hughes', 'bkr', 'oilfield services', 'energy', 'drilling',
        'oil and gas', 'equipment', 'lorenzo simonelli'
    },
    'MELI.US': {
        'mercado libre', 'meli', 'e-commerce', 'latin america', 'marketplace',
        'marcos galperin', 'online retail', 'mercadopago'
    },
    'AEP.US': {
        'american electric power', 'aep', 'utilities', 'electricity', 'power',
        'energy', 'transmission', 'distribution'
    },
    'GEHC.US': {
        'ge healthcare', 'gehc', 'medical imaging', 'healthcare technology',
        'mri', 'ct', 'ultrasound', 'peter arduini'
    },
    'ON.US': {
        'on semiconductor', 'on', 'onsemi', 'power management', 'sensors',
        'automotive', 'industrial', 'chips'
    },
    'TTD.US': {
        'trade desk', 'ttd', 'programmatic advertising', 'adtech', 'dsp',
        'jeff green', 'digital advertising', 'ctv'
    },
    'GFS.US': {
        'globalfoundries', 'gfs', 'semiconductor', 'foundry', 'chips',
        'manufacturing', 'fab', 'thomas caulfield'
    },
    'CTSH.US': {
        'cognizant', 'ctsh', 'it services', 'consulting', 'outsourcing',
        'digital transformation', 'ravi kumar'
    },
    'DASH.US': {
        'doordash', 'dash', 'food delivery', 'restaurant', 'logistics',
        'tony xu', 'on-demand', 'delivery'
    },
    'ROST.US': {
        'ross stores', 'rost', 'ross', 'off-price retail', 'discount',
        'apparel', 'home goods', 'barbara rentler'
    },
    'XEL.US': {
        'xcel energy', 'xel', 'utilities', 'electricity', 'natural gas',
        'renewable energy', 'power', 'bob frenzel'
    },
    'DLTR.US': {
        'dollar tree', 'dltr', 'dollar store', 'discount retail', 'family dollar',
        'value', 'retail', 'rick dreiling'
    },
    'WBD.US': {
        'warner bros discovery', 'wbd', 'hbo', 'max', 'warner bros', 'discovery',
        'streaming', 'david zaslav', 'entertainment'
    },
    'EA.US': {
        'electronic arts', 'ea', 'gaming', 'video games', 'fifa', 'madden',
        'apex legends', 'andrew wilson', 'entertainment software'
    },
    'ENPH.US': {
        'enphase', 'enph', 'solar', 'microinverters', 'energy storage',
        'renewable energy', 'badri kothandaraman'
    },
    'VRSK.US': {
        'verisk', 'vrsk', 'data analytics', 'insurance', 'risk assessment',
        'actuarial', 'underwriting', 'scott stephenson'
    },
    'CSGP.US': {
        'costar group', 'csgp', 'commercial real estate', 'apartments.com',
        'loopnet', 'andrew florance', 'property data'
    },
    'TTWO.US': {
        'take-two', 'ttwo', 'gaming', 'video games', 'rockstar', 'gta',
        '2k games', 'strauss zelnick', 'entertainment software'
    },
    'ALGN.US': {
        'align technology', 'algn', 'invisalign', 'dental', 'orthodontics',
        'clear aligners', 'joe hogan'
    },
    'EBAY.US': {
        'ebay', 'marketplace', 'online auction', 'e-commerce', 'jamie iannone',
        'selling', 'buying', 'online retail'
    },
    'ANET.US': {
        'arista networks', 'anet', 'networking', 'data center', 'cloud networking',
        'jayshree ullal', 'ethernet switches'
    },
    'FANG.US': {
        'diamondback energy', 'fang', 'oil', 'natural gas', 'permian basin',
        'upstream', 'exploration', 'production', 'travis stice'
    },
    'LCID.US': {
        'lucid', 'lcid', 'electric vehicles', 'ev', 'luxury', 'peter rawlinson',
        'lucid air', 'automotive'
    },
    'RIVN.US': {
        'rivian', 'rivn', 'electric vehicles', 'ev', 'trucks', 'suvs',
        'rj scaringe', 'automotive', 'amazon'
    },
    'MCHP.US': {
        'microchip', 'mchp', 'microcontrollers', 'semiconductors', 'embedded',
        'chips', 'sanghi', 'electronics'
    },
    'LULU.US': {
        'lululemon', 'lulu', 'athletic apparel', 'yoga', 'activewear',
        'calvin mcdonald', 'retail', 'fitness'
    },
    'WDAY.US': {
        'workday', 'wday', 'hr software', 'human resources', 'hcm',
        'financial management', 'aneel bhusri', 'enterprise software'
    },
    'CPRT.US': {
        'copart', 'cprt', 'online auctions', 'salvage', 'vehicles',
        'insurance', 'jay adair'
    },
    'MNST.US': {
        'monster beverage', 'mnst', 'energy drinks', 'beverages', 'monster',
        'rodney sacks', 'celsius'
    },
    'CHRW.US': {
        'c.h. robinson', 'chrw', 'logistics', 'freight', 'transportation',
        'supply chain', 'bob bistritzky'
    },
    'CEG.US': {
        'constellation energy', 'ceg', 'nuclear', 'renewable', 'power',
        'electricity', 'joe dominguez'
    },
    'MDB.US': {
        'mongodb', 'mdb', 'database', 'nosql', 'dev ittycheria',
        'developer tools', 'atlas', 'cloud database'
    },
    'PANW.US': {
        'palo alto networks', 'panw', 'cybersecurity', 'firewall', 'next-gen',
        'nikesh arora', 'cloud security', 'zero trust'
    },
    'DDOG.US': {
        'datadog', 'ddog', 'monitoring', 'observability', 'apm', 'infrastructure',
        'olivier pomel', 'devops', 'cloud monitoring'
    },
    'NET.US': {
        'cloudflare', 'net', 'cdn', 'ddos protection', 'web security',
        'matthew prince', 'internet infrastructure', 'waf'
    },
    'OKTA.US': {
        'okta', 'identity', 'authentication', 'sso', 'access management',
        'todd mckinnon', 'zero trust', 'iam'
    },
    'DOCN.US': {
        'digitalocean', 'docn', 'cloud', 'hosting', 'vps', 'developers',
        'yancey spruill', 'infrastructure'
    },
    'ESTC.US': {
        'elastic', 'estc', 'search', 'elasticsearch', 'kibana', 'logstash',
        'ash kulkarni', 'observability', 'security'
    },
    'SPLK.US': {
        'splunk', 'data analytics', 'siem', 'security', 'observability',
        'gary steele', 'machine data'
    },
    'NOW.US': {
        'servicenow', 'now', 'it service management', 'workflow', 'automation',
        'bill mcdermott', 'enterprise software', 'itsm'
    },
    'SNOW.US': {
        'snowflake', 'snow', 'data warehouse', 'cloud', 'analytics',
        'frank slootman', 'data platform', 'sql'
    },
    'PLTR.US': {
        'palantir', 'pltr', 'data analytics', 'government', 'defense',
        'alex karp', 'big data', 'platform'
    },
    'RBLX.US': {
        'roblox', 'rblx', 'gaming', 'metaverse', 'virtual worlds',
        'david baszucki', 'user-generated content', 'kids'
    },
}

# Company name mappings for fallback (ticker -> company name)
TICKER_COMPANY_NAMES = {
    'AAPL.US': 'apple',
    'MSFT.US': 'microsoft',
    'AMZN.US': 'amazon',
    'NVDA.US': 'nvidia',
    'GOOGL.US': 'google',
    'GOOG.US': 'google',
    'META.US': 'meta',
    'TSLA.US': 'tesla',
    'AVGO.US': 'broadcom',
    'COST.US': 'costco',
    'NFLX.US': 'netflix',
    'AMD.US': 'amd',
    'ADBE.US': 'adobe',
    'CSCO.US': 'cisco',
    'CMCSA.US': 'comcast',
    'INTC.US': 'intel',
    'TXN.US': 'texas instruments',
    'AMGN.US': 'amgen',
    'QCOM.US': 'qualcomm',
    'INTU.US': 'intuit',
    'ISRG.US': 'intuitive surgical',
    'VRTX.US': 'vertex',
    'BKNG.US': 'booking',
    'ADP.US': 'adp',
    'REGN.US': 'regeneron',
    'CDNS.US': 'cadence',
    'SNPS.US': 'synopsys',
    'CRWD.US': 'crowdstrike',
    'MRVL.US': 'marvell',
    'KLAC.US': 'kla',
    'NXPI.US': 'nxp',
    'CDW.US': 'cdw',
    'FTNT.US': 'fortinet',
    'ODFL.US': 'old dominion',
    'CTAS.US': 'cintas',
    'ANSS.US': 'ansys',
    'TEAM.US': 'atlassian',
    'FAST.US': 'fastenal',
    'PCAR.US': 'paccar',
    'EXPD.US': 'expeditors',
    'IDXX.US': 'idexx',
    'DXCM.US': 'dexcom',
    'ZS.US': 'zscaler',
    'BKR.US': 'baker hughes',
    'MELI.US': 'mercado libre',
    'AEP.US': 'american electric power',
    'GEHC.US': 'ge healthcare',
    'ON.US': 'on semiconductor',
    'TTD.US': 'trade desk',
    'GFS.US': 'globalfoundries',
    'CTSH.US': 'cognizant',
    'DASH.US': 'doordash',
    'ROST.US': 'ross stores',
    'XEL.US': 'xcel energy',
    'DLTR.US': 'dollar tree',
    'WBD.US': 'warner bros discovery',
    'EA.US': 'electronic arts',
    'ENPH.US': 'enphase',
    'VRSK.US': 'verisk',
    'CSGP.US': 'costar',
    'TTWO.US': 'take-two',
    'ALGN.US': 'align',
    'EBAY.US': 'ebay',
    'ANET.US': 'arista',
    'FANG.US': 'diamondback',
    'LCID.US': 'lucid',
    'RIVN.US': 'rivian',
    'MCHP.US': 'microchip',
    'LULU.US': 'lululemon',
    'WDAY.US': 'workday',
    'CPRT.US': 'copart',
    'MNST.US': 'monster',
    'CHRW.US': 'c.h. robinson',
    'CEG.US': 'constellation',
    'MDB.US': 'mongodb',
    'PANW.US': 'palo alto',
    'DDOG.US': 'datadog',
    'NET.US': 'cloudflare',
    'OKTA.US': 'okta',
    'DOCN.US': 'digitalocean',
    'ESTC.US': 'elastic',
    'SPLK.US': 'splunk',
    'NOW.US': 'servicenow',
    'SNOW.US': 'snowflake',
    'PLTR.US': 'palantir',
    'RBLX.US': 'roblox',
}


def get_ticker_keywords(ticker: str) -> Set[str]:
    """
    Get keywords for a ticker symbol.
    
    Args:
        ticker: Ticker symbol (e.g., 'AAPL.US')
        
    Returns:
        Set of keywords for the ticker, or empty set if not found
    """
    return TICKER_KEYWORDS.get(ticker.upper(), set())


def get_ticker_fallback_keywords(ticker: str) -> Set[str]:
    """
    Get fallback keywords for a ticker (ticker symbol and company name).
    
    Args:
        ticker: Ticker symbol (e.g., 'AAPL.US')
        
    Returns:
        Set of fallback keywords (ticker symbol and company name)
    """
    ticker_upper = ticker.upper()
    keywords = {ticker_upper.replace('.US', '')}  # Add ticker symbol without .US
    
    # Add company name if available
    company_name = TICKER_COMPANY_NAMES.get(ticker_upper)
    if company_name:
        keywords.add(company_name.lower())
    
    return keywords


def is_ticker_relevant(title: str, content: str, ticker: str, threshold: int = 2) -> bool:
    """
    Check if article is relevant to a ticker based on keywords.
    
    Uses ticker-specific keywords if available, otherwise falls back to
    ticker symbol and company name matching with a lower threshold.
    
    Args:
        title: Article title
        content: Article content (first 500 chars checked)
        ticker: Ticker symbol (e.g., 'AAPL.US')
        threshold: Minimum number of keyword matches
        
    Returns:
        True if article appears relevant to the ticker
    """
    if not title and not content:
        return False
    
    text = f"{title} {content[:500]}".lower()
    ticker_upper = ticker.upper()
    
    # Get ticker-specific keywords
    keywords = get_ticker_keywords(ticker_upper)
    
    # If we have specific keywords, use them
    if keywords:
        matches = sum(1 for keyword in keywords if keyword in text)
        return matches >= threshold
    else:
        # Fallback: use ticker symbol and company name with lower threshold
        fallback_keywords = get_ticker_fallback_keywords(ticker_upper)
        matches = sum(1 for keyword in fallback_keywords if keyword.lower() in text)
        # Use threshold of 1 for fallback (less strict)
        return matches >= 1


def filter_news_articles(
    articles: List[Dict],
    ticker: str = 'AAPL.US',
    filters: Dict = None
) -> tuple:
    """
    Filter news articles to keep only relevant ones.
    
    Args:
        articles: List of article dictionaries
        ticker: Ticker symbol to filter for
        filters: Dictionary with filter settings:
            - max_symbols: Maximum number of symbols (default: 10)
            - require_primary: Ticker must be first symbol (default: False)
            - require_keywords: Must have ticker-specific keywords (default: True)
            - min_title_length: Minimum title length (default: 10)
            - min_content_length: Minimum content length (default: 50)
            - keyword_threshold: Min keyword matches (default: 2)
            - exclude_tags: Tags to exclude (default: [])
            
    Returns:
        Tuple of (filtered_articles, filter_stats)
    """
    if filters is None:
        filters = {}
    
    # Default filter settings
    max_symbols = filters.get('max_symbols', 10)
    require_primary = filters.get('require_primary', False)
    require_keywords = filters.get('require_keywords', True)
    min_title_length = filters.get('min_title_length', 10)
    min_content_length = filters.get('min_content_length', 50)
    keyword_threshold = filters.get('keyword_threshold', 2)
    exclude_tags = set(filters.get('exclude_tags', []))
    
    filtered = []
    stats = {
        'total': len(articles),
        'filtered_out': 0,
        'reasons': Counter()
    }
    
    for article in articles:
        symbols = article.get('symbols', [])
        title = article.get('title', '') or ''
        content = article.get('content', '') or ''
        tags = set(article.get('tags', []))
        
        # Check if ticker is in symbols
        if ticker not in symbols:
            stats['filtered_out'] += 1
            stats['reasons']['ticker_not_in_symbols'] += 1
            continue
        
        # Filter 1: Exclude articles with excluded tags
        if exclude_tags and tags.intersection(exclude_tags):
            stats['filtered_out'] += 1
            stats['reasons']['excluded_tag'] += 1
            continue
        
        # Filter 2: Too many symbols (likely generic market news)
        if len(symbols) > max_symbols:
            stats['filtered_out'] += 1
            stats['reasons'][f'too_many_symbols_{len(symbols)}'] += 1
            continue
        
        # Filter 3: Require ticker to be primary (first symbol)
        if require_primary and symbols[0] != ticker:
            stats['filtered_out'] += 1
            stats['reasons']['not_primary_symbol'] += 1
            continue
        
        # Filter 4: Minimum content length
        if len(title) < min_title_length or len(content) < min_content_length:
            stats['filtered_out'] += 1
            stats['reasons']['too_short'] += 1
            continue
        
        # Filter 5: Keyword relevance (most important)
        if require_keywords and not is_ticker_relevant(title, content, ticker, keyword_threshold):
            stats['filtered_out'] += 1
            stats['reasons']['no_keywords'] += 1
            continue
        
        # Article passed all filters
        filtered.append(article)
    
    stats['kept'] = len(filtered)
    stats['retention_rate'] = (len(filtered) / len(articles) * 100) if articles else 0
    
    return filtered, stats


def analyze_filtering_impact(
    articles: List[Dict],
    ticker: str = 'AAPL.US',
    filter_configs: List[Dict] = None
) -> None:
    """
    Analyze impact of different filter configurations.
    
    Args:
        articles: List of article dictionaries
        ticker: Ticker symbol
        filter_configs: List of filter configurations to test
    """
    if filter_configs is None:
        filter_configs = [
            {
                'name': 'Baseline (no filters)',
                'filters': {
                    'max_symbols': 1000,  # Effectively no limit
                    'require_primary': False,
                    'require_keywords': False,
                }
            },
            {
                'name': 'Keyword filter only',
                'filters': {
                    'max_symbols': 1000,
                    'require_primary': False,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
            {
                'name': 'Max 5 symbols + keywords',
                'filters': {
                    'max_symbols': 5,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
            {
                'name': 'Max 3 symbols + keywords',
                'filters': {
                    'max_symbols': 3,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
            {
                'name': 'Primary symbol + keywords',
                'filters': {
                    'max_symbols': 10,
                    'require_primary': True,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
            {
                'name': 'Strict: Primary + max 3 symbols + keywords',
                'filters': {
                    'max_symbols': 3,
                    'require_primary': True,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
        ]
    
    print(f"\n{'='*80}")
    print(f"FILTERING ANALYSIS FOR {ticker}")
    print(f"{'='*80}")
    print(f"Total articles: {len(articles)}\n")
    
    for config in filter_configs:
        filtered, stats = filter_news_articles(articles, ticker, config['filters'])
        
        print(f"{config['name']}:")
        print(f"  Kept: {stats['kept']} ({stats['retention_rate']:.1f}%)")
        print(f"  Filtered out: {stats['filtered_out']}")
        
        if stats['reasons']:
            print(f"  Top reasons for filtering:")
            for reason, count in stats['reasons'].most_common(5):
                print(f"    - {reason}: {count}")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter news articles to keep only relevant ones for a ticker'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to JSON file with news data')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    parser.add_argument('--ticker', type=str, default='AAPL.US',
                       help='Ticker symbol to filter for')
    parser.add_argument('--max-symbols', type=int, default=10,
                       help='Maximum number of symbols per article')
    parser.add_argument('--require-primary', action='store_true',
                       help='Require ticker to be first symbol')
    parser.add_argument('--require-keywords', action='store_true', default=True,
                       help='Require ticker-specific keywords in title/content')
    parser.add_argument('--no-keywords', dest='require_keywords', action='store_false',
                       help='Disable keyword requirement')
    parser.add_argument('--keyword-threshold', type=int, default=2,
                       help='Minimum keyword matches required')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze different filter configurations')
    
    args = parser.parse_args()
    
    # Load articles
    json_file = Path(args.input)
    if not json_file.exists():
        print(f"Error: File not found: {args.input}")
        return 1
    
    print(f"Loading articles from {args.input}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles")
    
    # Analyze different filter configurations
    if args.analyze:
        analyze_filtering_impact(articles, args.ticker)
        return 0
    
    # Apply filters
    filters = {
        'max_symbols': args.max_symbols,
        'require_primary': args.require_primary,
        'require_keywords': args.require_keywords,
        'keyword_threshold': args.keyword_threshold,
    }
    
    filtered, stats = filter_news_articles(articles, args.ticker, filters)
    
    print(f"\n{'='*60}")
    print("FILTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total articles: {stats['total']}")
    print(f"Kept: {stats['kept']} ({stats['retention_rate']:.1f}%)")
    print(f"Filtered out: {stats['filtered_out']}")
    print(f"\nTop reasons for filtering:")
    for reason, count in stats['reasons'].most_common(10):
        print(f"  {reason}: {count}")
    
    # Save filtered articles
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved {len(filtered)} filtered articles to {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


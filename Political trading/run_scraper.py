#!/usr/bin/env python3
"""
Capitol Trades Enhanced Scraper Runner
Easy-to-use script for scraping congressional trading data
"""

import argparse
import sys
from enhanced_scraper import CapitolTradesScraper

def main():
    parser = argparse.ArgumentParser(description='Enhanced Capitol Trades Scraper')
    parser.add_argument('--max-trades', type=int, default=500, 
                        help='Maximum number of trades to scrape (default: 500)')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run browser in headless mode (default: True)')
    parser.add_argument('--visible', action='store_true', 
                        help='Run browser in visible mode (for debugging)')
    parser.add_argument('--output', type=str, default='exports/capitol_trades.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Set headless mode
    headless = not args.visible
    
    print("="*60)
    print("🏛️  CAPITOL TRADES ENHANCED SCRAPER 🏛️")
    print("="*60)
    print(f"📊 Max trades: {args.max_trades}")
    print(f"🖥️  Headless: {headless}")
    print(f"📁 Output: {args.output}")
    print("="*60)
    
    # Initialize scraper
    scraper = CapitolTradesScraper(headless=headless)
    
    try:
        # Scrape trades
        print("\n🔍 Starting scrape...")
        trades = scraper.scrape_trades(max_trades=args.max_trades)
        
        if trades:
            print(f"✅ Successfully scraped {len(trades)} trades")
            
            # Export data
            scraper.export_data(trades, args.output)
            
            # Display analysis
            analysis = scraper.analyze_trades(trades)
            print("\n" + "="*50)
            print("📊 INVESTMENT ANALYSIS:")
            print("="*50)
            print(f"📈 Total Trades: {analysis['total_trades']}")
            print(f"🔥 Recent Activity (30 days): {analysis['recent_activity']}")
            
            if analysis['buy_sell_ratio']:
                print("\n📊 BUY/SELL BREAKDOWN:")
                for action, count in analysis['buy_sell_ratio'].items():
                    print(f"  {action}: {count}")
            
            if analysis['most_active_stocks']:
                print("\n🎯 MOST ACTIVE STOCKS:")
                for ticker, count in list(analysis['most_active_stocks'].items())[:5]:
                    print(f"  {ticker}: {count} trades")
            
            if analysis['party_breakdown']:
                print("\n🏛️ PARTY BREAKDOWN:")
                for party, count in analysis['party_breakdown'].items():
                    print(f"  {party}: {count}")
            
            if analysis['recent_big_trades']:
                print(f"\n💰 RECENT BIG TRADES (>${50000}+):")
                for trade in analysis['recent_big_trades'][:3]:
                    print(f"  {trade.get('ticker', 'N/A')} - {trade.get('action', 'N/A')} - ${trade.get('amount_min', 'N/A')}")
            
            print("="*50)
            print("✅ Scraping completed successfully!")
            print("📁 Check the 'exports/' directory for detailed data and analysis.")
            
        else:
            print("❌ No trades found. This might be due to:")
            print("   - Website structure changes")
            print("   - Rate limiting")
            print("   - Network issues")
            print("   - Check debug files for more information")
            
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        print("Try running with --visible to see what's happening in the browser")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 
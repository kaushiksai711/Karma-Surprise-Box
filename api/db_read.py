import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import typer
from pathlib import Path

app = typer.Typer()

def get_db_connection(db_path: str = 'karma_rewards.db'):
    """Create and return a database connection."""
    return sqlite3.connect(db_path)

def get_all_rewards(db_path: str = 'karma_rewards.db') -> List[Dict[str, Any]]:
    """Retrieve all reward records from the database."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, user_id, box_type, timestamp 
            FROM rewarded_users 
            ORDER BY date DESC, timestamp DESC
        """)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

def get_rewards_by_date(date: str, db_path: str = 'karma_rewards.db') -> List[Dict[str, Any]]:
    """Retrieve rewards for a specific date."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, user_id, box_type, timestamp 
            FROM rewarded_users 
            WHERE date = ?
            ORDER BY timestamp DESC
        """, (date,))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

def get_rewards_by_user(user_id: str, db_path: str = 'karma_rewards.db') -> List[Dict[str, Any]]:
    """Retrieve all rewards for a specific user."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, user_id, box_type, timestamp 
            FROM rewarded_users 
            WHERE user_id = ?
            ORDER BY date DESC, timestamp DESC
        """, (user_id,))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

def display_rewards(rewards: List[Dict[str, Any]], title: str = "Rewards"):
    """Display rewards in a formatted table."""
    if not rewards:
        print(f"No {title.lower()} found.")
        return
    
    df = pd.DataFrame(rewards)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{title}:")
    print("-" * 60)
    print(df.to_string(index=False))
    print(f"\nTotal: {len(rewards)} records")

@app.command()
def show_all():
    """Show all reward records."""
    rewards = get_all_rewards()
    display_rewards(rewards, "All Reward Records")

@app.command()
def by_date(date: str):
    """Show rewards for a specific date (YYYY-MM-DD)."""
    try:
        # Validate date format
        datetime.strptime(date, '%Y-%m-%d')
        rewards = get_rewards_by_date(date)
        display_rewards(rewards, f"Rewards for {date}")
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD")

@app.command()
def by_user(user_id: str):
    """Show all rewards for a specific user."""
    rewards = get_rewards_by_user(user_id)
    display_rewards(rewards, f"Rewards for User: {user_id}")

@app.command()
def summary():
    """Show summary statistics of rewards."""
    with get_db_connection() as conn:
        # Total rewards
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rewarded_users")
        total_rewards = cursor.fetchone()[0]
        
        # Unique users
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM rewarded_users")
        unique_users = cursor.fetchone()[0]
        
        # Rewards by box type
        cursor.execute("""
            SELECT box_type, COUNT(*) as count 
            FROM rewarded_users 
            GROUP BY box_type 
            ORDER BY count DESC
        """)
        box_type_counts = dict(cursor.fetchall())
        
        # Recent rewards
        cursor.execute("""
            SELECT date, COUNT(*) as count 
            FROM rewarded_users 
            GROUP BY date 
            ORDER BY date DESC 
            LIMIT 7
        """)
        recent_rewards = dict(cursor.fetchall())
    
    print("\n=== Reward Statistics ===")
    print(f"Total rewards issued: {total_rewards}")
    print(f"Unique users rewarded: {unique_users}")
    
    print("\nRewards by box type:")
    for box_type, count in box_type_counts.items():
        print(f"  - {box_type}: {count}")
    
    print("\nRecent rewards (last 7 days):")
    for date, count in recent_rewards.items():
        print(f"  - {date}: {count} rewards")

if __name__ == "__main__":
    app()
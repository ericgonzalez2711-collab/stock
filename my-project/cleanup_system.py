#!/usr/bin/env python3
"""
Automated cleanup system for the algo-trading repository.
Removes unnecessary files and maintains repository hygiene.
"""

import os
import shutil
import glob
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse


class RepositoryCleanup:
    """Handles automated cleanup of unnecessary files in the trading repository."""
    
    def __init__(self, dry_run: bool = True):
        """
        Initialize cleanup system.
        
        Args:
            dry_run: If True, only shows what would be deleted without actually deleting
        """
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent
        self.deleted_files = []
        self.deleted_dirs = []
        self.space_saved = 0
        
        print(f"🧹 Repository Cleanup System")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE CLEANUP'}")
        print(f"Project Root: {self.project_root}")
        print("=" * 50)
    
    def clean_python_cache(self):
        """Remove Python cache files and __pycache__ directories."""
        print("\n🐍 Cleaning Python Cache Files...")
        
        # Find all __pycache__ directories
        pycache_dirs = list(self.project_root.rglob("__pycache__"))
        
        # Find all .pyc files
        pyc_files = list(self.project_root.rglob("*.pyc"))
        pyo_files = list(self.project_root.rglob("*.pyo"))
        
        all_cache_files = pyc_files + pyo_files
        
        print(f"Found {len(pycache_dirs)} __pycache__ directories")
        print(f"Found {len(all_cache_files)} .pyc/.pyo files")
        
        # Remove .pyc/.pyo files
        for file_path in all_cache_files:
            size = file_path.stat().st_size if file_path.exists() else 0
            self.space_saved += size
            
            if self.dry_run:
                print(f"  Would delete: {file_path}")
            else:
                try:
                    file_path.unlink()
                    self.deleted_files.append(str(file_path))
                    print(f"  ✅ Deleted: {file_path}")
                except Exception as e:
                    print(f"  ❌ Failed to delete {file_path}: {e}")
        
        # Remove __pycache__ directories
        for dir_path in pycache_dirs:
            try:
                if dir_path.exists():
                    # Calculate directory size
                    dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    self.space_saved += dir_size
                    
                    if self.dry_run:
                        print(f"  Would delete directory: {dir_path}")
                    else:
                        shutil.rmtree(dir_path)
                        self.deleted_dirs.append(str(dir_path))
                        print(f"  ✅ Deleted directory: {dir_path}")
            except Exception as e:
                print(f"  ❌ Failed to delete directory {dir_path}: {e}")
    
    def clean_log_files(self, max_age_days: int = 7, max_size_mb: int = 10):
        """
        Clean old or large log files.
        
        Args:
            max_age_days: Delete logs older than this many days
            max_size_mb: Archive logs larger than this size
        """
        print(f"\n📄 Cleaning Log Files (older than {max_age_days} days or larger than {max_size_mb}MB)...")
        
        logs_dir = self.project_root / "logs"
        if not logs_dir.exists():
            print("  No logs directory found")
            return
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for log_file in logs_dir.glob("*.log"):
            file_stat = log_file.stat()
            file_age = file_stat.st_mtime
            file_size = file_stat.st_size
            
            should_clean = False
            reason = ""
            
            if file_age < cutoff_time:
                should_clean = True
                reason = f"older than {max_age_days} days"
            elif file_size > max_size_bytes:
                should_clean = True
                reason = f"larger than {max_size_mb}MB ({file_size/1024/1024:.1f}MB)"
            
            if should_clean:
                self.space_saved += file_size
                
                if self.dry_run:
                    print(f"  Would clean: {log_file} ({reason})")
                else:
                    try:
                        # Archive large files, delete old files
                        if file_size > max_size_bytes:
                            archive_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d')}.log.old"
                            archive_path = log_file.parent / archive_name
                            log_file.rename(archive_path)
                            print(f"  📦 Archived: {log_file} → {archive_name}")
                        else:
                            log_file.unlink()
                            print(f"  ✅ Deleted: {log_file} ({reason})")
                        
                        self.deleted_files.append(str(log_file))
                    except Exception as e:
                        print(f"  ❌ Failed to clean {log_file}: {e}")
    
    def clean_cache_data(self, max_age_days: int = 7):
        """
        Clean old cached market data.
        
        Args:
            max_age_days: Delete cache files older than this many days
        """
        print(f"\n💾 Cleaning Cache Data (older than {max_age_days} days)...")
        
        cache_dir = self.project_root / "cache"
        if not cache_dir.exists():
            print("  No cache directory found")
            return
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        for cache_file in cache_dir.glob("*.json"):
            file_stat = cache_file.stat()
            file_age = file_stat.st_mtime
            file_size = file_stat.st_size
            
            if file_age < cutoff_time:
                self.space_saved += file_size
                
                if self.dry_run:
                    print(f"  Would delete: {cache_file} ({file_size/1024:.1f}KB, {(time.time()-file_age)/86400:.1f} days old)")
                else:
                    try:
                        cache_file.unlink()
                        self.deleted_files.append(str(cache_file))
                        print(f"  ✅ Deleted: {cache_file} ({file_size/1024:.1f}KB)")
                    except Exception as e:
                        print(f"  ❌ Failed to delete {cache_file}: {e}")
    
    def clean_debug_output(self):
        """Remove debug output files."""
        print("\n🐛 Cleaning Debug Output Files...")
        
        debug_dir = self.project_root / "debug_sheets_output"
        if not debug_dir.exists():
            print("  No debug output directory found")
            return
        
        for debug_file in debug_dir.glob("*.json"):
            file_size = debug_file.stat().st_size
            self.space_saved += file_size
            
            if self.dry_run:
                print(f"  Would delete: {debug_file} ({file_size} bytes)")
            else:
                try:
                    debug_file.unlink()
                    self.deleted_files.append(str(debug_file))
                    print(f"  ✅ Deleted: {debug_file}")
                except Exception as e:
                    print(f"  ❌ Failed to delete {debug_file}: {e}")
    
    def clean_temporary_files(self):
        """Remove temporary and test files."""
        print("\n🧪 Cleaning Temporary and Test Files...")
        
        # Test and demo scripts (created for demonstration)
        temp_files = [
            "test_improved_strategy.py",
            "setup_and_test.py", 
            "simple_strategy_demo.py"
        ]
        
        for filename in temp_files:
            file_path = self.project_root / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                self.space_saved += file_size
                
                if self.dry_run:
                    print(f"  Would delete: {filename} ({file_size/1024:.1f}KB)")
                else:
                    try:
                        file_path.unlink()
                        self.deleted_files.append(str(file_path))
                        print(f"  ✅ Deleted: {filename}")
                    except Exception as e:
                        print(f"  ❌ Failed to delete {filename}: {e}")
    
    def clean_assignment_documents(self):
        """Remove assignment documents (keep source code only)."""
        print("\n📄 Cleaning Assignment Documents...")
        
        doc_files = [
            "Automation_Assignment_Updated.docx"
        ]
        
        for filename in doc_files:
            file_path = self.project_root / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                self.space_saved += file_size
                
                if self.dry_run:
                    print(f"  Would delete: {filename} ({file_size/1024:.1f}KB)")
                else:
                    try:
                        file_path.unlink()
                        self.deleted_files.append(str(file_path))
                        print(f"  ✅ Deleted: {filename}")
                    except Exception as e:
                        print(f"  ❌ Failed to delete {filename}: {e}")
    
    def clean_empty_directories(self):
        """Remove empty directories."""
        print("\n📁 Cleaning Empty Directories...")
        
        # Find all directories and check if empty
        for dir_path in self.project_root.rglob("*"):
            if dir_path.is_dir() and dir_path.name not in ['.git', '.', '..']:
                try:
                    # Check if directory is empty
                    if not any(dir_path.iterdir()):
                        if self.dry_run:
                            print(f"  Would delete empty directory: {dir_path}")
                        else:
                            dir_path.rmdir()
                            self.deleted_dirs.append(str(dir_path))
                            print(f"  ✅ Deleted empty directory: {dir_path}")
                except Exception as e:
                    # Directory not empty or other error
                    pass
    
    def create_gitignore(self):
        """Create or update .gitignore to prevent future clutter."""
        print("\n📝 Creating/Updating .gitignore...")
        
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/*.log
*.log

# Cache and temporary files
cache/
*.tmp
*.temp
temp/
tmp/

# Debug output
debug_*/
debug_output/
debug_sheets_output/

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Environment variables
.env
.env.local
.env.*.local

# Model files (optional - uncomment if models are large)
# models/*.pkl
# models/*.joblib

# Data files (optional - uncomment if you don't want to track data)
# data/
# *.csv
# *.xlsx

# Assignment and documentation (optional)
*.docx
*.pdf
assignment*
Assignment*

# Test and demo files
test_*.py
*_test.py
*_demo.py
demo_*.py
setup_and_test.py
simple_*.py
"""
        
        gitignore_path = self.project_root / ".gitignore"
        
        if self.dry_run:
            print(f"  Would create/update: {gitignore_path}")
            print("  Content preview:")
            print("    " + "\n    ".join(gitignore_content.split('\n')[:10]) + "\n    ...")
        else:
            try:
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content)
                print(f"  ✅ Created/Updated: {gitignore_path}")
            except Exception as e:
                print(f"  ❌ Failed to create .gitignore: {e}")
    
    def run_full_cleanup(self, include_cache: bool = True, include_logs: bool = True):
        """
        Run complete cleanup process.
        
        Args:
            include_cache: Whether to clean cache files
            include_logs: Whether to clean log files
        """
        print(f"\n🚀 Starting {'DRY RUN' if self.dry_run else 'LIVE'} Cleanup...")
        start_time = time.time()
        
        # Clean Python cache (always safe)
        self.clean_python_cache()
        
        # Clean temporary test files
        self.clean_temporary_files()
        
        # Clean debug output
        self.clean_debug_output()
        
        # Clean assignment documents
        self.clean_assignment_documents()
        
        # Clean logs if requested
        if include_logs:
            self.clean_log_files()
        
        # Clean cache if requested
        if include_cache:
            self.clean_cache_data()
        
        # Clean empty directories
        self.clean_empty_directories()
        
        # Create/update .gitignore
        self.create_gitignore()
        
        # Summary
        elapsed_time = time.time() - start_time
        self.print_summary(elapsed_time)
    
    def print_summary(self, elapsed_time: float):
        """Print cleanup summary."""
        print("\n" + "=" * 50)
        print("🧹 CLEANUP SUMMARY")
        print("=" * 50)
        
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE CLEANUP'}")
        print(f"Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"Files {'Would Be ' if self.dry_run else ''}Deleted: {len(self.deleted_files)}")
        print(f"Directories {'Would Be ' if self.dry_run else ''}Deleted: {len(self.deleted_dirs)}")
        print(f"Space {'Would Be ' if self.dry_run else ''}Saved: {self.space_saved / 1024 / 1024:.2f} MB")
        
        if not self.dry_run and (self.deleted_files or self.deleted_dirs):
            print("\n✅ Cleanup completed successfully!")
            print("📊 Repository is now cleaner and more organized.")
        elif self.dry_run:
            print(f"\n💡 Run with --live flag to actually perform cleanup")
        else:
            print(f"\n✨ Repository was already clean!")


def create_automated_cleanup_script():
    """Create a script that can be scheduled to run automatically."""
    script_content = """#!/bin/bash
# Automated cleanup script for algo-trading repository
# Can be added to crontab for regular maintenance

cd "$(dirname "$0")"

echo "🧹 Running automated repository cleanup..."
echo "Timestamp: $(date)"

# Run cleanup (live mode, preserve recent cache and logs)
python3 cleanup_system.py --live --preserve-recent-cache --preserve-recent-logs

echo "✅ Automated cleanup completed"
echo "----------------------------------------"
"""
    
    script_path = Path(__file__).parent / "auto_cleanup.sh"
    
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"✅ Created automated cleanup script: {script_path}")
        print("💡 To schedule daily cleanup, add to crontab:")
        print(f"   0 2 * * * {script_path}")  # Run at 2 AM daily
        
    except Exception as e:
        print(f"❌ Failed to create automated script: {e}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Repository Cleanup System for Algo-Trading Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_system.py                    # Dry run (show what would be deleted)
  python cleanup_system.py --live             # Actually delete files
  python cleanup_system.py --live --no-cache  # Keep cache files
  python cleanup_system.py --live --no-logs   # Keep log files
  python cleanup_system.py --create-scheduler # Create automated cleanup script
        """
    )
    
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Actually delete files (default is dry run)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true', 
        help='Skip cleaning cache files'
    )
    
    parser.add_argument(
        '--no-logs',
        action='store_true',
        help='Skip cleaning log files'
    )
    
    parser.add_argument(
        '--preserve-recent-cache',
        action='store_true',
        help='Only delete cache files older than 7 days'
    )
    
    parser.add_argument(
        '--preserve-recent-logs', 
        action='store_true',
        help='Only delete log files older than 7 days or larger than 10MB'
    )
    
    parser.add_argument(
        '--create-scheduler',
        action='store_true',
        help='Create automated cleanup script for scheduling'
    )
    
    args = parser.parse_args()
    
    if args.create_scheduler:
        create_automated_cleanup_script()
        return
    
    # Initialize cleanup system
    cleanup = RepositoryCleanup(dry_run=not args.live)
    
    # Run cleanup
    cleanup.run_full_cleanup(
        include_cache=not args.no_cache,
        include_logs=not args.no_logs
    )
    
    if args.live:
        print(f"\n🎉 Repository cleanup completed!")
        print(f"💾 Space saved: {cleanup.space_saved / 1024 / 1024:.2f} MB")
    else:
        print(f"\n💡 This was a dry run. Use --live flag to actually delete files.")
        print(f"💾 Potential space savings: {cleanup.space_saved / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
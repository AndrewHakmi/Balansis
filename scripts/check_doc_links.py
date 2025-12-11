#!/usr/bin/env python3
"""
Script to check documentation links for validity.
Validates internal links, external URLs, and cross-references.
"""

import os
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json


class DocumentationLinkChecker:
    """Check documentation links for validity."""
    
    def __init__(self, docs_dir: Path, timeout: int = 10):
        self.docs_dir = docs_dir
        self.timeout = timeout
        self.internal_links: Set[str] = set()
        self.external_links: Set[str] = set()
        self.broken_links: List[Tuple[str, str, str]] = []
        self.warnings: List[Tuple[str, str, str]] = []
        
    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files in documentation directory."""
        md_files = []
        for root, dirs, files in os.walk(self.docs_dir):
            # Skip hidden directories and build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '_build']
            for file in files:
                if file.endswith('.md'):
                    md_files.append(Path(root) / file)
        return md_files
    
    def extract_links(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """Extract internal and external links from markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            self.warnings.append((str(file_path), "encoding", "Could not read file with UTF-8 encoding"))
            return [], []
        
        # Regex patterns for different link types
        markdown_links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', content)
        reference_links = re.findall(r'\[([^\]]*)\]:\s*([^\s]+)', content)
        html_links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>', content)
        
        internal_links = []
        external_links = []
        
        all_links = [link[1] for link in markdown_links + reference_links] + html_links
        
        for link in all_links:
            # Skip anchors and fragments without base URL
            if link.startswith('#'):
                continue
                
            # Skip email links
            if link.startswith('mailto:'):
                continue
                
            # Skip javascript links
            if link.startswith('javascript:'):
                continue
                
            # External links
            if link.startswith(('http://', 'https://')):
                external_links.append(link)
            # Internal links
            elif not link.startswith(('ftp://', 'file://')):
                internal_links.append(link)
                
        return internal_links, external_links
    
    def check_internal_link(self, file_path: Path, link: str) -> bool:
        """Check if internal link exists."""
        # Handle relative paths
        if link.startswith('./'):
            link = link[2:]
        elif link.startswith('../'):
            # Handle parent directory references
            current_dir = file_path.parent
            while link.startswith('../'):
                link = link[3:]
                current_dir = current_dir.parent
            target_path = current_dir / link
        else:
            # Relative to current file's directory
            target_path = file_path.parent / link
            
        # Remove fragment identifier
        if '#' in link:
            link_path, fragment = link.split('#', 1)
            target_path = file_path.parent / link_path if link_path else file_path
        else:
            target_path = file_path.parent / link
            
        # Resolve path
        try:
            target_path = target_path.resolve()
        except (OSError, ValueError):
            return False
            
        # Check if file exists
        if target_path.exists():
            return True
            
        # Check if it's a directory with index file
        if target_path.is_dir():
            index_files = ['index.md', 'README.md', 'index.html']
            for index_file in index_files:
                if (target_path / index_file).exists():
                    return True
                    
        return False
    
    def check_external_link(self, url: str) -> Tuple[bool, str]:
        """Check if external URL is accessible."""
        try:
            # Add user agent to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; DocumentationLinkChecker/1.0)'
            }
            response = requests.head(url, timeout=self.timeout, headers=headers, allow_redirects=True)
            
            # Some servers don't support HEAD requests
            if response.status_code == 405:
                response = requests.get(url, timeout=self.timeout, headers=headers, allow_redirects=True)
                
            if response.status_code == 200:
                return True, "OK"
            elif 300 <= response.status_code < 400:
                return True, f"Redirect ({response.status_code})"
            else:
                return False, f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "Timeout"
        except requests.exceptions.ConnectionError:
            return False, "Connection Error"
        except requests.exceptions.RequestException as e:
            return False, f"Request Error: {str(e)}"
        except Exception as e:
            return False, f"Unknown Error: {str(e)}"
    
    def check_all_links(self, max_workers: int = 10) -> Dict:
        """Check all links in documentation."""
        md_files = self.find_markdown_files()
        
        print(f"Found {len(md_files)} markdown files")
        
        # Collect all links
        all_internal_links = []
        all_external_links = []
        
        for file_path in md_files:
            internal, external = self.extract_links(file_path)
            for link in internal:
                all_internal_links.append((file_path, link))
                self.internal_links.add(link)
            for link in external:
                all_external_links.append((file_path, link))
                self.external_links.add(link)
        
        print(f"Found {len(all_internal_links)} internal links and {len(all_external_links)} external links")
        
        # Check internal links
        print("Checking internal links...")
        for file_path, link in all_internal_links:
            if not self.check_internal_link(file_path, link):
                self.broken_links.append((str(file_path), link, "Internal link not found"))
        
        # Check external links with threading
        print("Checking external links...")
        unique_external_links = list(self.external_links)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.check_external_link, url): url 
                for url in unique_external_links
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    is_valid, message = future.result()
                    if not is_valid:
                        # Find all files that reference this broken link
                        for file_path, link in all_external_links:
                            if link == url:
                                self.broken_links.append((str(file_path), link, message))
                except Exception as e:
                    # Find all files that reference this problematic link
                    for file_path, link in all_external_links:
                        if link == url:
                            self.broken_links.append((str(file_path), link, f"Check failed: {str(e)}"))
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive report."""
        report = {
            "summary": {
                "total_internal_links": len(self.internal_links),
                "total_external_links": len(self.external_links),
                "broken_links": len(self.broken_links),
                "warnings": len(self.warnings),
                "status": "PASS" if len(self.broken_links) == 0 else "FAIL"
            },
            "broken_links": [
                {"file": file, "link": link, "error": error}
                for file, link, error in self.broken_links
            ],
            "warnings": [
                {"file": file, "type": warning_type, "message": message}
                for file, warning_type, message in self.warnings
            ]
        }
        
        return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check documentation links")
    parser.add_argument(
        "--docs-dir", 
        type=Path, 
        default=Path("docs"),
        help="Documentation directory (default: docs)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=10,
        help="Timeout for external links in seconds (default: 10)"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=10,
        help="Maximum number of worker threads (default: 10)"
    )
    parser.add_argument(
        "--output", 
        type=Path,
        help="Output file for JSON report"
    )
    parser.add_argument(
        "--fail-on-warnings", 
        action="store_true",
        help="Fail if warnings are found"
    )
    
    args = parser.parse_args()
    
    if not args.docs_dir.exists():
        print(f"Error: Documentation directory '{args.docs_dir}' does not exist")
        sys.exit(1)
    
    checker = DocumentationLinkChecker(args.docs_dir, args.timeout)
    report = checker.check_all_links(args.max_workers)
    
    # Print summary
    print("\n" + "="*50)
    print("LINK CHECK SUMMARY")
    print("="*50)
    print(f"Internal links: {report['summary']['total_internal_links']}")
    print(f"External links: {report['summary']['total_external_links']}")
    print(f"Broken links: {report['summary']['broken_links']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Status: {report['summary']['status']}")
    
    # Print broken links
    if report['broken_links']:
        print("\nBROKEN LINKS:")
        print("-" * 30)
        for broken in report['broken_links']:
            print(f"File: {broken['file']}")
            print(f"Link: {broken['link']}")
            print(f"Error: {broken['error']}")
            print()
    
    # Print warnings
    if report['warnings']:
        print("\nWARNINGS:")
        print("-" * 30)
        for warning in report['warnings']:
            print(f"File: {warning['file']}")
            print(f"Type: {warning['type']}")
            print(f"Message: {warning['message']}")
            print()
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.output}")
    
    # Exit with appropriate code
    if report['summary']['status'] == "FAIL":
        sys.exit(1)
    elif args.fail_on_warnings and report['summary']['warnings'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script to validate API documentation completeness.
Ensures all public classes and methods have proper documentation.
"""

import ast
import inspect
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import argparse
import json


class APIDocumentationValidator:
    """Validate API documentation completeness."""
    
    def __init__(self, package_dir: Path, docs_dir: Path):
        self.package_dir = package_dir
        self.docs_dir = docs_dir
        self.missing_docs: List[Dict] = []
        self.incomplete_docs: List[Dict] = []
        self.warnings: List[Dict] = []
        
    def get_public_modules(self) -> List[str]:
        """Get list of public modules in the package."""
        modules = []
        
        # Add package root to Python path
        sys.path.insert(0, str(self.package_dir.parent))
        
        try:
            # Import the main package
            package_name = self.package_dir.name
            package = importlib.import_module(package_name)
            
            # Get all public modules from __all__ if available
            if hasattr(package, '__all__'):
                for item in package.__all__:
                    try:
                        module = getattr(package, item)
                        if inspect.ismodule(module):
                            modules.append(f"{package_name}.{item}")
                        elif inspect.isclass(module):
                            # For classes, get their module
                            modules.append(module.__module__)
                    except AttributeError:
                        continue
            
            # Also scan for Python files
            for py_file in self.package_dir.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                # Convert file path to module name
                rel_path = py_file.relative_to(self.package_dir.parent)
                module_name = str(rel_path.with_suffix("")).replace("/", ".").replace("\\", ".")
                
                if not module_name.startswith("_"):  # Skip private modules
                    modules.append(module_name)
                    
        except ImportError as e:
            self.warnings.append({
                "type": "import_error",
                "message": f"Could not import package: {e}"
            })
            
        return list(set(modules))  # Remove duplicates
    
    def analyze_module(self, module_name: str) -> Dict:
        """Analyze a module for public classes and functions."""
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            self.warnings.append({
                "type": "module_import_error",
                "module": module_name,
                "message": str(e)
            })
            return {}
            
        analysis = {
            "module": module_name,
            "classes": [],
            "functions": [],
            "constants": []
        }
        
        # Get all public members
        if hasattr(module, '__all__'):
            public_names = module.__all__
        else:
            public_names = [name for name in dir(module) if not name.startswith('_')]
        
        for name in public_names:
            try:
                obj = getattr(module, name)
                
                if inspect.isclass(obj):
                    class_info = self.analyze_class(obj)
                    analysis["classes"].append(class_info)
                elif inspect.isfunction(obj):
                    func_info = self.analyze_function(obj)
                    analysis["functions"].append(func_info)
                elif not inspect.ismodule(obj) and not callable(obj):
                    # Constants
                    analysis["constants"].append({
                        "name": name,
                        "type": type(obj).__name__,
                        "docstring": getattr(obj, '__doc__', None)
                    })
                    
            except Exception as e:
                self.warnings.append({
                    "type": "member_analysis_error",
                    "module": module_name,
                    "member": name,
                    "message": str(e)
                })
                
        return analysis
    
    def analyze_class(self, cls) -> Dict:
        """Analyze a class for documentation completeness."""
        class_info = {
            "name": cls.__name__,
            "docstring": inspect.getdoc(cls),
            "methods": [],
            "properties": [],
            "class_variables": []
        }
        
        # Analyze methods
        for name, method in inspect.getmembers(cls, inspect.ismethod):
            if not name.startswith('_') or name in ['__init__', '__str__', '__repr__']:
                method_info = self.analyze_function(method)
                class_info["methods"].append(method_info)
        
        # Analyze functions (unbound methods)
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith('_') or name in ['__init__', '__str__', '__repr__']:
                func_info = self.analyze_function(func)
                class_info["methods"].append(func_info)
        
        # Analyze properties
        for name, prop in inspect.getmembers(cls, lambda x: isinstance(x, property)):
            if not name.startswith('_'):
                class_info["properties"].append({
                    "name": name,
                    "docstring": inspect.getdoc(prop),
                    "getter": prop.fget is not None,
                    "setter": prop.fset is not None,
                    "deleter": prop.fdel is not None
                })
        
        return class_info
    
    def analyze_function(self, func) -> Dict:
        """Analyze a function for documentation completeness."""
        try:
            signature = inspect.signature(func)
        except (ValueError, TypeError):
            signature = None
            
        func_info = {
            "name": func.__name__,
            "docstring": inspect.getdoc(func),
            "signature": str(signature) if signature else None,
            "parameters": [],
            "returns": None
        }
        
        if signature:
            for param_name, param in signature.parameters.items():
                param_info = {
                    "name": param_name,
                    "annotation": str(param.annotation) if param.annotation != param.empty else None,
                    "default": str(param.default) if param.default != param.empty else None,
                    "kind": str(param.kind)
                }
                func_info["parameters"].append(param_info)
                
            if signature.return_annotation != signature.empty:
                func_info["returns"] = str(signature.return_annotation)
        
        return func_info
    
    def check_documentation_exists(self, module_analysis: Dict) -> List[Dict]:
        """Check if documentation files exist for the analyzed module."""
        missing = []
        module_name = module_analysis["module"]
        
        # Expected documentation paths
        module_parts = module_name.split('.')
        if len(module_parts) > 1:
            # Remove package name
            doc_path = self.docs_dir / "api" / "/".join(module_parts[1:])
        else:
            doc_path = self.docs_dir / "api" / module_parts[0]
            
        doc_file = doc_path.with_suffix('.md')
        
        if not doc_file.exists():
            missing.append({
                "type": "missing_module_doc",
                "module": module_name,
                "expected_path": str(doc_file)
            })
        
        # Check class documentation
        for class_info in module_analysis["classes"]:
            class_doc_file = self.docs_dir / "api" / f"{class_info['name'].lower()}.md"
            if not class_doc_file.exists():
                missing.append({
                    "type": "missing_class_doc",
                    "module": module_name,
                    "class": class_info["name"],
                    "expected_path": str(class_doc_file)
                })
        
        return missing
    
    def check_docstring_quality(self, analysis: Dict) -> List[Dict]:
        """Check quality of docstrings."""
        issues = []
        module_name = analysis["module"]
        
        # Check module docstring
        if not analysis.get("docstring"):
            issues.append({
                "type": "missing_module_docstring",
                "module": module_name
            })
        
        # Check class docstrings
        for class_info in analysis["classes"]:
            if not class_info["docstring"]:
                issues.append({
                    "type": "missing_class_docstring",
                    "module": module_name,
                    "class": class_info["name"]
                })
            
            # Check method docstrings
            for method in class_info["methods"]:
                if not method["docstring"] and method["name"] != "__init__":
                    issues.append({
                        "type": "missing_method_docstring",
                        "module": module_name,
                        "class": class_info["name"],
                        "method": method["name"]
                    })
        
        # Check function docstrings
        for func_info in analysis["functions"]:
            if not func_info["docstring"]:
                issues.append({
                    "type": "missing_function_docstring",
                    "module": module_name,
                    "function": func_info["name"]
                })
        
        return issues
    
    def validate_all(self) -> Dict:
        """Validate all API documentation."""
        modules = self.get_public_modules()
        
        print(f"Found {len(modules)} modules to validate")
        
        all_analyses = []
        
        for module_name in modules:
            print(f"Analyzing module: {module_name}")
            analysis = self.analyze_module(module_name)
            
            if analysis:  # Skip empty analyses
                all_analyses.append(analysis)
                
                # Check for missing documentation files
                missing_docs = self.check_documentation_exists(analysis)
                self.missing_docs.extend(missing_docs)
                
                # Check docstring quality
                docstring_issues = self.check_docstring_quality(analysis)
                self.incomplete_docs.extend(docstring_issues)
        
        return self.generate_report(all_analyses)
    
    def generate_report(self, analyses: List[Dict]) -> Dict:
        """Generate validation report."""
        total_classes = sum(len(a["classes"]) for a in analyses)
        total_functions = sum(len(a["functions"]) for a in analyses)
        total_methods = sum(
            len(c["methods"]) for a in analyses for c in a["classes"]
        )
        
        report = {
            "summary": {
                "modules_analyzed": len(analyses),
                "total_classes": total_classes,
                "total_functions": total_functions,
                "total_methods": total_methods,
                "missing_docs": len(self.missing_docs),
                "incomplete_docs": len(self.incomplete_docs),
                "warnings": len(self.warnings),
                "status": "PASS" if len(self.missing_docs) == 0 and len(self.incomplete_docs) == 0 else "FAIL"
            },
            "missing_documentation": self.missing_docs,
            "incomplete_documentation": self.incomplete_docs,
            "warnings": self.warnings,
            "detailed_analysis": analyses
        }
        
        return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate API documentation completeness")
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("balansis"),
        help="Package directory to analyze (default: balansis)"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Documentation directory (default: docs)"
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
    
    if not args.package_dir.exists():
        print(f"Error: Package directory '{args.package_dir}' does not exist")
        sys.exit(1)
        
    if not args.docs_dir.exists():
        print(f"Error: Documentation directory '{args.docs_dir}' does not exist")
        sys.exit(1)
    
    validator = APIDocumentationValidator(args.package_dir, args.docs_dir)
    report = validator.validate_all()
    
    # Print summary
    print("\n" + "="*50)
    print("API DOCUMENTATION VALIDATION SUMMARY")
    print("="*50)
    print(f"Modules analyzed: {report['summary']['modules_analyzed']}")
    print(f"Total classes: {report['summary']['total_classes']}")
    print(f"Total functions: {report['summary']['total_functions']}")
    print(f"Total methods: {report['summary']['total_methods']}")
    print(f"Missing documentation: {report['summary']['missing_docs']}")
    print(f"Incomplete documentation: {report['summary']['incomplete_docs']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Status: {report['summary']['status']}")
    
    # Print missing documentation
    if report['missing_documentation']:
        print("\nMISSING DOCUMENTATION:")
        print("-" * 30)
        for missing in report['missing_documentation']:
            print(f"Type: {missing['type']}")
            if 'module' in missing:
                print(f"Module: {missing['module']}")
            if 'class' in missing:
                print(f"Class: {missing['class']}")
            if 'expected_path' in missing:
                print(f"Expected: {missing['expected_path']}")
            print()
    
    # Print incomplete documentation
    if report['incomplete_documentation']:
        print("\nINCOMPLETE DOCUMENTATION:")
        print("-" * 30)
        for incomplete in report['incomplete_documentation']:
            print(f"Type: {incomplete['type']}")
            if 'module' in incomplete:
                print(f"Module: {incomplete['module']}")
            if 'class' in incomplete:
                print(f"Class: {incomplete['class']}")
            if 'method' in incomplete:
                print(f"Method: {incomplete['method']}")
            if 'function' in incomplete:
                print(f"Function: {incomplete['function']}")
            print()
    
    # Print warnings
    if report['warnings']:
        print("\nWARNINGS:")
        print("-" * 30)
        for warning in report['warnings']:
            print(f"Type: {warning['type']}")
            print(f"Message: {warning['message']}")
            if 'module' in warning:
                print(f"Module: {warning['module']}")
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
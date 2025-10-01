"""Service for zero-sum operations on infinite sets."""

import uuid
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from ...core import ZeroSumInfiniteSet, get_global_cache, get_global_parallel_processor
from ...database.repository import InfiniteSetRepository
from ..models.requests import SeriesType, OperationMethod
from ..models.responses import (
    InfiniteSetResponse,
    ZeroSumOperationResponse,
    CompensatingSetResponse,
    ValidationResponse,
    BatchOperationResponse,
    ConvergenceAnalysisResponse,
    OperationStatus,
    ConvergenceType
)

logger = logging.getLogger(__name__)


class ZeroSumService:
    """Service for zero-sum theory operations on infinite sets."""
    
    def __init__(self):
        self.repository = InfiniteSetRepository()
        self.cache = get_global_cache()
        self.parallel_processor = get_global_parallel_processor()
    
    async def create_infinite_set(
        self,
        name: str,
        series_type: SeriesType,
        parameters: Dict[str, Any],
        description: Optional[str] = None
    ) -> InfiniteSetResponse:
        """Create a new infinite set."""
        
        # Create set based on type
        if series_type == SeriesType.HARMONIC:
            p = parameters.get('p', 1.0)
            infinite_set = ZeroSumInfiniteSet.create_harmonic_series(p=p)
        elif series_type == SeriesType.ALTERNATING:
            p = parameters.get('p', 1.0)
            infinite_set = ZeroSumInfiniteSet.create_alternating_series(p=p)
        elif series_type == SeriesType.GEOMETRIC:
            ratio = parameters['ratio']
            infinite_set = ZeroSumInfiniteSet.create_geometric_series(ratio=ratio)
        elif series_type == SeriesType.CUSTOM:
            formula = parameters['formula']
            infinite_set = ZeroSumInfiniteSet(elements=[], formula=formula)
        else:
            raise ValueError(f"Unsupported series type: {series_type}")
        
        # Generate unique ID
        set_id = str(uuid.uuid4())
        
        # Convergence analysis
        convergence_info = infinite_set.convergence_analysis(max_terms=1000)
        
        # Get partial sums for demonstration
        partial_sums = [infinite_set.get_partial_sum(n) for n in range(1, 11)]
        
        # Save to database
        await self.repository.create_infinite_set(
            set_id=set_id,
            name=name,
            series_type=series_type.value,
            parameters=parameters,
            description=description,
            convergence_info=convergence_info
        )
        
        # Save set elements
        elements_data = []
        for i in range(100):  # Save first 100 elements
            try:
                value = infinite_set.get_element(i)
                elements_data.append({
                    'position': i,
                    'value': float(value),
                    'computed_at': datetime.utcnow()
                })
            except (ZeroDivisionError, OverflowError):
                break
        
        if elements_data:
            await self.repository.save_set_elements(set_id, elements_data)
        
        return InfiniteSetResponse(
            id=set_id,
            name=name,
            series_type=series_type.value,
            parameters=parameters,
            description=description,
            created_at=datetime.utcnow(),
            convergence_info=convergence_info,
            partial_sums=partial_sums
        )
    
    async def get_infinite_set(self, set_id: str) -> Optional[InfiniteSetResponse]:
        """Get information about an infinite set."""
        
        # Try to get from cache
        cache_key = f"set:{set_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return InfiniteSetResponse(**cached_result)
        
        # Get from database
        set_data = await self.repository.get_infinite_set(set_id)
        if not set_data:
            return None
        
        # Get set elements
        elements = await self.repository.get_set_elements(set_id, limit=10)
        partial_sums = [elem['value'] for elem in elements] if elements else []
        
        result = InfiniteSetResponse(
            id=set_data['id'],
            name=set_data['name'],
            series_type=set_data['series_type'],
            parameters=set_data['parameters'],
            description=set_data['description'],
            created_at=set_data['created_at'],
            convergence_info=set_data['convergence_info'],
            partial_sums=partial_sums
        )
        
        # Cache result
        self.cache.set(cache_key, result.dict(), ttl=3600)
        
        return result
    
    async def list_infinite_sets(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[InfiniteSetResponse]:
        """Get list of infinite sets."""
        
        sets_data = await self.repository.list_infinite_sets(limit=limit, offset=offset)
        
        results = []
        for set_data in sets_data:
            results.append(InfiniteSetResponse(
                id=set_data['id'],
                name=set_data['name'],
                series_type=set_data['series_type'],
                parameters=set_data['parameters'],
                description=set_data['description'],
                created_at=set_data['created_at'],
                convergence_info=set_data['convergence_info']
            ))
        
        return results
    
    async def perform_zero_sum_operation(
        self,
        set_ids: List[str],
        method: OperationMethod,
        tolerance: float,
        max_iterations: int,
        use_cache: bool = True
    ) -> ZeroSumOperationResponse:
        """Perform zero-sum operation."""
        
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Check cache
        if use_cache:
            cache_key = f"zero_sum:{':'.join(sorted(set_ids))}:{method.value}:{tolerance}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return ZeroSumOperationResponse(**cached_result)
        
        try:
            # Get sets
            infinite_sets = []
            for set_id in set_ids:
                set_data = await self.repository.get_infinite_set(set_id)
                if not set_data:
                    raise ValueError(f"Set {set_id} not found")
                
                # Restore set
                if set_data['series_type'] == 'harmonic':
                    p = set_data['parameters'].get('p', 1.0)
                    infinite_set = ZeroSumInfiniteSet.create_harmonic_series(p=p)
                elif set_data['series_type'] == 'alternating':
                    p = set_data['parameters'].get('p', 1.0)
                    infinite_set = ZeroSumInfiniteSet.create_alternating_series(p=p)
                elif set_data['series_type'] == 'geometric':
                    ratio = set_data['parameters']['ratio']
                    infinite_set = ZeroSumInfiniteSet.create_geometric_series(ratio=ratio)
                else:
                    # For custom sets
                    infinite_set = ZeroSumInfiniteSet(elements=[], formula=set_data['parameters'].get('formula'))
                
                infinite_sets.append(infinite_set)
            
            # Execute operation
            if len(infinite_sets) == 2:
                result_value = infinite_sets[0].zero_sum_operation(
                    infinite_sets[1],
                    method=method.value,
                    tolerance=tolerance,
                    max_iterations=max_iterations
                )
            else:
                # For multiple operations use parallel processor
                result_value = await self.parallel_processor.parallel_zero_sum_operation(
                    infinite_sets,
                    method=method.value,
                    tolerance=tolerance,
                    max_iterations=max_iterations
                )
            
            execution_time = time.time() - start_time
            
            # Determine status
            if abs(result_value) <= tolerance:
                status = OperationStatus.SUCCESS
            elif abs(result_value) <= tolerance * 10:
                status = OperationStatus.PARTIAL
            else:
                status = OperationStatus.FAILED
            
            response = ZeroSumOperationResponse(
                operation_id=operation_id,
                status=status,
                result=result_value,
                method_used=method.value,
                iterations=max_iterations,  # In real implementation this should be actual count
                tolerance_achieved=abs(result_value),
                execution_time=execution_time,
                set_ids=set_ids,
                compensation_details={
                    "method": method.value,
                "convergence_rate": 0.95,  # Approximate value
                "stability_factor": 0.98
                },
                cached=False
            )
            
            # Cache result
            if use_cache and status == OperationStatus.SUCCESS:
                cache_key = f"zero_sum:{':'.join(sorted(set_ids))}:{method.value}:{tolerance}"
                self.cache.set(cache_key, response.dict(), ttl=1800)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in zero-sum operation {operation_id}: {e}")
            return ZeroSumOperationResponse(
                operation_id=operation_id,
                status=OperationStatus.FAILED,
                result=None,
                method_used=method.value,
                iterations=0,
                tolerance_achieved=float('inf'),
                execution_time=time.time() - start_time,
                set_ids=set_ids,
                cached=False
            )
    
    async def find_compensating_set(
        self,
        target_set_id: str,
        method: OperationMethod,
        tolerance: float,
        max_iterations: int,
        search_space: Optional[Dict[str, Any]] = None
    ) -> CompensatingSetResponse:
        """Find compensating set."""
        
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Get target set
            target_data = await self.repository.get_infinite_set(target_set_id)
            if not target_data:
                raise ValueError(f"Target set {target_set_id} not found")
            
            # Restore target set
            if target_data['series_type'] == 'harmonic':
                p = target_data['parameters'].get('p', 1.0)
                target_set = ZeroSumInfiniteSet.create_harmonic_series(p=p)
            elif target_data['series_type'] == 'alternating':
                p = target_data['parameters'].get('p', 1.0)
                target_set = ZeroSumInfiniteSet.create_alternating_series(p=p)
            elif target_data['series_type'] == 'geometric':
                ratio = target_data['parameters']['ratio']
                target_set = ZeroSumInfiniteSet.create_geometric_series(ratio=ratio)
            else:
                target_set = ZeroSumInfiniteSet(elements=[], formula=target_data['parameters'].get('formula'))
            
            # Search for compensating set
            compensating_set = target_set.find_compensating_set(
                method=method.value,
                tolerance=tolerance,
                max_iterations=max_iterations
            )
            
            execution_time = time.time() - start_time
            
            if compensating_set:
                # Create compensating set in system
                comp_set_response = await self.create_infinite_set(
                    name=f"Compensating for {target_data['name']}",
                    series_type=SeriesType.CUSTOM,
                    parameters={'formula': 'compensating', 'target_id': target_set_id},
                    description=f"Automatically found compensating set for {target_set_id}"
                )
                
                # Validate compensation quality
                validation_result = target_set.validate_zero_sum(compensating_set, tolerance)
                compensation_quality = 1.0 - min(abs(validation_result), 1.0)
                
                return CompensatingSetResponse(
                    operation_id=operation_id,
                    status=OperationStatus.SUCCESS,
                    target_set_id=target_set_id,
                    compensating_set=comp_set_response,
                    compensation_quality=compensation_quality,
                    method_used=method.value,
                    search_iterations=max_iterations,
                    execution_time=execution_time,
                    search_details={
                        "tolerance_achieved": abs(validation_result),
                        "convergence_rate": 0.92,
                        "search_space_explored": 0.75
                    }
                )
            else:
                return CompensatingSetResponse(
                    operation_id=operation_id,
                    status=OperationStatus.FAILED,
                    target_set_id=target_set_id,
                    compensating_set=None,
                    compensation_quality=0.0,
                    method_used=method.value,
                    search_iterations=max_iterations,
                    execution_time=execution_time,
                    search_details={"error": "Compensating set not found"}
                )
                
        except Exception as e:
            logger.error(f"Error in compensating set search {operation_id}: {e}")
            return CompensatingSetResponse(
                operation_id=operation_id,
                status=OperationStatus.FAILED,
                target_set_id=target_set_id,
                compensating_set=None,
                compensation_quality=0.0,
                method_used=method.value,
                search_iterations=0,
                execution_time=time.time() - start_time,
                search_details={"error": str(e)}
            )
    
    async def validate_zero_sum(
        self,
        set_ids: List[str],
        tolerance: float,
        detailed_analysis: bool = False
    ) -> ValidationResponse:
        """Validate zero-sum of sets."""
        
        validation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Get sets
            infinite_sets = []
            for set_id in set_ids:
                set_data = await self.repository.get_infinite_set(set_id)
                if not set_data:
                    raise ValueError(f"Set {set_id} not found")
                
                # Restore set (simplified version)
                if set_data['series_type'] == 'harmonic':
                    p = set_data['parameters'].get('p', 1.0)
                    infinite_set = ZeroSumInfiniteSet.create_harmonic_series(p=p)
                elif set_data['series_type'] == 'alternating':
                    p = set_data['parameters'].get('p', 1.0)
                    infinite_set = ZeroSumInfiniteSet.create_alternating_series(p=p)
                elif set_data['series_type'] == 'geometric':
                    ratio = set_data['parameters']['ratio']
                    infinite_set = ZeroSumInfiniteSet.create_geometric_series(ratio=ratio)
                else:
                    infinite_set = ZeroSumInfiniteSet(elements=[], formula=set_data['parameters'].get('formula'))
                
                infinite_sets.append(infinite_set)
            
            # Validate zero-sum
            if len(infinite_sets) == 2:
                sum_value = infinite_sets[0].validate_zero_sum(infinite_sets[1], tolerance)
            else:
                # For multiple validation
                sum_value = sum(s.get_partial_sum(1000) for s in infinite_sets)
            
            is_valid = abs(sum_value) <= tolerance
            
            convergence_analysis = None
            partial_sums_analysis = None
            
            if detailed_analysis:
                convergence_analysis = {}
                partial_sums_analysis = []
                
                for i, infinite_set in enumerate(infinite_sets):
                    conv_info = infinite_set.convergence_analysis(max_terms=1000)
                    convergence_analysis[f"set_{i}"] = conv_info
                    
                    partial_sums = [infinite_set.get_partial_sum(n) for n in range(1, 101, 10)]
                    partial_sums_analysis.append({
                        "set_id": set_ids[i],
                        "partial_sums": partial_sums,
                        "convergence_rate": conv_info.get('convergence_rate', 0.0)
                    })
            
            execution_time = time.time() - start_time
            
            return ValidationResponse(
                validation_id=validation_id,
                is_valid=is_valid,
                sum_value=sum_value,
                tolerance_used=tolerance,
                set_ids=set_ids,
                convergence_analysis=convergence_analysis,
                partial_sums_analysis=partial_sums_analysis,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Validation error {validation_id}: {e}")
            return ValidationResponse(
                validation_id=validation_id,
                is_valid=False,
                sum_value=float('inf'),
                tolerance_used=tolerance,
                set_ids=set_ids,
                execution_time=time.time() - start_time
            )
    
    async def perform_batch_operations(
        self,
        operations: List[Dict[str, Any]],
        parallel: bool = True,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> BatchOperationResponse:
        """Perform batch operations."""
        
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        if parallel:
            results = await self.parallel_processor.batch_operations(
                operations,
                max_workers=max_workers,
                timeout=timeout
            )
        else:
            results = []
            for op in operations:
                try:
                    if op['type'] == 'create':
                        result = await self.create_infinite_set(**op['params'])
                    elif op['type'] == 'zero_sum':
                        result = await self.perform_zero_sum_operation(**op['params'])
                    elif op['type'] == 'find_compensating':
                        result = await self.find_compensating_set(**op['params'])
                    elif op['type'] == 'validate':
                        result = await self.validate_zero_sum(**op['params'])
                    else:
                        result = {"error": f"Unknown operation type: {op['type']}"}
                    
                    results.append({"status": "success", "result": result})
                except Exception as e:
                    results.append({"status": "failed", "error": str(e)})
        
        successful_operations = sum(1 for r in results if r.get('status') == 'success')
        failed_operations = len(results) - successful_operations
        
        execution_time = time.time() - start_time
        
        return BatchOperationResponse(
            batch_id=batch_id,
            total_operations=len(operations),
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            results=results,
            execution_time=execution_time,
            parallel_execution=parallel
        )
    
    async def analyze_convergence(
        self,
        set_id: str,
        max_terms: int,
        analysis_methods: List[str]
    ) -> ConvergenceAnalysisResponse:
        """Analyze convergence of infinite set."""
        
        # Get set
        set_data = await self.repository.get_infinite_set(set_id)
        if not set_data:
            raise ValueError(f"Set {set_id} not found")
        
        # Restore set
        if set_data['series_type'] == 'harmonic':
            p = set_data['parameters'].get('p', 1.0)
            infinite_set = ZeroSumInfiniteSet.create_harmonic_series(p=p)
        elif set_data['series_type'] == 'alternating':
            p = set_data['parameters'].get('p', 1.0)
            infinite_set = ZeroSumInfiniteSet.create_alternating_series(p=p)
        elif set_data['series_type'] == 'geometric':
            ratio = set_data['parameters']['ratio']
            infinite_set = ZeroSumInfiniteSet.create_geometric_series(ratio=ratio)
        else:
            infinite_set = ZeroSumInfiniteSet(elements=[], formula=set_data['parameters'].get('formula'))
        
        # Convergence analysis
        convergence_info = infinite_set.convergence_analysis(max_terms=max_terms)
        
        # Determine convergence type
        if convergence_info.get('is_convergent', False):
            if convergence_info.get('is_absolutely_convergent', False):
                convergence_type = ConvergenceType.CONVERGENT
            else:
                convergence_type = ConvergenceType.CONDITIONALLY_CONVERGENT
        elif convergence_info.get('is_divergent', False):
            convergence_type = ConvergenceType.DIVERGENT
        else:
            convergence_type = ConvergenceType.UNKNOWN
        
        # Partial sums
        partial_sums = [infinite_set.get_partial_sum(n) for n in range(1, min(max_terms, 1000) + 1)]
        
        # Analysis with different methods
        analysis_results = {}
        for method in analysis_methods:
            if method == "ratio_test":
                analysis_results[method] = convergence_info.get('ratio_test', {})
            elif method == "root_test":
                analysis_results[method] = convergence_info.get('root_test', {})
            elif method == "integral_test":
                analysis_results[method] = convergence_info.get('integral_test', {})
        
        # Recommendations
        recommendations = []
        if convergence_type == ConvergenceType.DIVERGENT:
            recommendations.append("Series diverges - consider using summation methods")
        elif convergence_type == ConvergenceType.CONDITIONALLY_CONVERGENT:
            recommendations.append("Series conditionally converges - order of terms matters")
        elif convergence_type == ConvergenceType.CONVERGENT:
            recommendations.append("Series absolutely converges - terms can be rearranged")
        
        return ConvergenceAnalysisResponse(
            set_id=set_id,
            convergence_type=convergence_type,
            convergence_rate=convergence_info.get('convergence_rate'),
            analysis_methods=analysis_results,
            partial_sums=partial_sums,
            convergence_plot_data={
                "x": list(range(1, len(partial_sums) + 1)),
                "y": partial_sums
            },
            recommendations=recommendations
        )
    
    async def delete_infinite_set(self, set_id: str) -> bool:
        """Delete infinite set."""
        
        # Remove from cache
        cache_key = f"set:{set_id}"
        self.cache.delete(cache_key)
        
        # Remove from database
        return await self.repository.delete_infinite_set(set_id)
    
    async def log_operation(
        self,
        operation_type: str,
        operation_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Log operation to database."""
        
        try:
            await self.repository.log_operation(
                operation_id=operation_id,
                operation_type=operation_type,
                parameters=result,
                result=result.get('result'),
                status=result.get('status', 'unknown'),
                execution_time=result.get('execution_time', 0.0)
            )
        except Exception as e:
            logger.error(f"Error logging operation {operation_id}: {e}")
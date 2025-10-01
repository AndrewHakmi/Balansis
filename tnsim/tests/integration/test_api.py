"""Integration tests for TNSIM FastAPI application."""

import pytest
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from tnsim.api.main import app
from tnsim.core.sets import ZeroSumInfiniteSet
from tnsim.api.services import ZeroSumService


class TestHealthEndpoints:
    """System health endpoint tests."""
    
    def test_health_check(self, test_client):
        """Test basic health check."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_api_info(self, test_client):
        """Test getting API information."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert data["name"] == "TNSIM API"


class TestZeroSumEndpoints:
    """Zero sum operation endpoint tests."""
    
    def test_create_zero_sum_set_basic(self, test_client):
        """Test creating a basic zero sum set."""
        payload = {
            "elements": [1, -1, 0.5, -0.5],
            "metadata": {
                "name": "test_set",
                "description": "Test zero sum set"
            }
        }
        
        response = test_client.post("/api/zerosum/sets", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["elements"] == payload["elements"]
        assert data["metadata"]["name"] == "test_set"
    
    def test_create_zero_sum_set_series_type(self, test_client):
        """Test creating a set with specified series type."""
        payload = {
            "series_type": "harmonic",
            "n_terms": 100,
            "metadata": {
                "name": "harmonic_series"
            }
        }
        
        response = test_client.post("/api/zerosum/sets", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert len(data["elements"]) == 100
        assert data["metadata"]["name"] == "harmonic_series"
    
    def test_create_zero_sum_set_invalid_data(self, test_client):
        """Test creating a set with invalid data."""
        # Empty elements and no series_type
        payload = {
            "metadata": {"name": "invalid_set"}
        }
        
        response = test_client.post("/api/zerosum/sets", json=payload)
        
        assert response.status_code == 422
    
    def test_get_zero_sum_set(self, test_client, sample_set_id):
        """Test getting a set by ID."""
        response = test_client.get(f"/api/zerosum/sets/{sample_set_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_set_id
        assert "elements" in data
        assert "metadata" in data
    
    def test_get_nonexistent_set(self, test_client):
        """Test getting a non-existent set."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = test_client.get(f"/api/zerosum/sets/{fake_id}")
        
        assert response.status_code == 404
    
    def test_list_zero_sum_sets(self, test_client):
        """Test getting a list of sets."""
        response = test_client.get("/api/zerosum/sets")
        
        assert response.status_code == 200
        data = response.json()
        assert "sets" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert isinstance(data["sets"], list)
    
    def test_list_sets_with_pagination(self, test_client):
        """Test pagination of set list."""
        response = test_client.get("/api/zerosum/sets?page=1&page_size=5")
        
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 5
        assert len(data["sets"]) <= 5
    
    def test_update_zero_sum_set(self, test_client, sample_set_id):
        """Test updating a set."""
        update_payload = {
            "metadata": {
                "name": "updated_set",
                "description": "Updated description"
            }
        }
        
        response = test_client.put(f"/api/zerosum/sets/{sample_set_id}", json=update_payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["name"] == "updated_set"
        assert data["metadata"]["description"] == "Updated description"
    
    def test_delete_zero_sum_set(self, test_client):
        """Test deleting a set."""
        # First create a set
        create_payload = {
            "elements": [1, 2, 3],
            "metadata": {"name": "to_delete"}
        }
        create_response = test_client.post("/api/zerosum/sets", json=create_payload)
        set_id = create_response.json()["id"]
        
        # Then delete it
        response = test_client.delete(f"/api/zerosum/sets/{set_id}")
        
        assert response.status_code == 204
        
        # Verify that the set is actually deleted
        get_response = test_client.get(f"/api/zerosum/sets/{set_id}")
        assert get_response.status_code == 404


class TestOperationEndpoints:
    """Operation endpoint tests."""
    
    def test_zero_sum_operation_direct(self, test_client, sample_set_id):
        """Test direct zero sum operation."""
        payload = {"method": "direct"}
        
        response = test_client.post(
            f"/api/zerosum/sets/{sample_set_id}/operations/zero-sum",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sum" in data
        assert "method" in data
        assert data["method"] == "direct"
        assert "execution_time" in data
    
    def test_zero_sum_operation_compensated(self, test_client, sample_set_id):
        """Test compensated zero sum operation."""
        payload = {"method": "compensated"}
        
        response = test_client.post(
            f"/api/zerosum/sets/{sample_set_id}/operations/zero-sum",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sum" in data
        assert data["method"] == "compensated"
        assert "compensation_terms" in data
    
    def test_zero_sum_operation_invalid_method(self, test_client, sample_set_id):
        """Test operation with invalid method."""
        payload = {"method": "invalid_method"}
        
        response = test_client.post(
            f"/api/zerosum/sets/{sample_set_id}/operations/zero-sum",
            json=payload
        )
        
        assert response.status_code == 422
    
    def test_find_compensating_set(self, test_client, sample_set_id):
        """Test finding compensating set."""
        payload = {
            "method": "iterative",
            "max_iterations": 100,
            "tolerance": 1e-10
        }
        
        response = test_client.post(
            f"/api/zerosum/sets/{sample_set_id}/operations/compensating-set",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "compensating_elements" in data
        assert "compensation_quality" in data
        assert "method" in data
        assert "iterations_used" in data
    
    def test_validate_zero_sum(self, test_client, sample_set_id):
        """Test zero sum validation."""
        payload = {
            "tolerance": 1e-12,
            "detailed": True
        }
        
        response = test_client.post(
            f"/api/zerosum/sets/{sample_set_id}/operations/validate",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_zero_sum" in data
        assert "total_sum" in data
        assert "tolerance_used" in data
        assert "validation_details" in data
    
    def test_convergence_analysis(self, test_client, sample_set_id):
        """Test convergence analysis."""
        payload = {
            "method": "ratio_test",
            "n_terms": 1000
        }
        
        response = test_client.post(
            f"/api/zerosum/sets/{sample_set_id}/operations/convergence",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_convergent" in data
        assert "convergence_type" in data
        assert "analysis_method" in data
        assert "convergence_rate" in data


class TestParallelEndpoints:
    """Parallel operation endpoint tests."""
    
    def test_parallel_zero_sum_operations(self, test_client):
        """Test parallel zero sum operations."""
        payload = {
            "set_ids": [],  # Will be filled in fixture
            "method": "direct",
            "max_workers": 2
        }
        
        # Create several sets for testing
        set_ids = []
        for i in range(3):
            create_payload = {
                "elements": [i+1, -(i+1), (i+1)*0.5, -(i+1)*0.5],
                "metadata": {"name": f"parallel_test_{i}"}
            }
            create_response = test_client.post("/api/zerosum/sets", json=create_payload)
            set_ids.append(create_response.json()["id"])
        
        payload["set_ids"] = set_ids
        
        response = test_client.post("/api/zerosum/parallel/zero-sum", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == len(set_ids)
        assert "execution_time" in data
        assert "parallel_efficiency" in data
    
    def test_batch_operations(self, test_client):
        """Test batch operations."""
        # Create set for testing
        create_payload = {
            "elements": [1, -1, 2, -2],
            "metadata": {"name": "batch_test"}
        }
        create_response = test_client.post("/api/zerosum/sets", json=create_payload)
        set_id = create_response.json()["id"]
        
        payload = {
            "operations": [
                {
                    "type": "zero_sum",
                    "set_id": set_id,
                    "method": "direct"
                },
                {
                    "type": "validation",
                    "set_id": set_id,
                    "tolerance": 1e-10
                },
                {
                    "type": "convergence_analysis",
                    "set_id": set_id,
                    "method": "ratio_test"
                }
            ],
            "max_workers": 2,
            "ignore_errors": True
        }
        
        response = test_client.post("/api/zerosum/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3
        assert "total_execution_time" in data
        assert "successful_operations" in data


class TestAnalyticsEndpoints:
    """Analytics endpoint tests."""
    
    def test_get_statistics(self, test_client):
        """Test getting statistics."""
        response = test_client.get("/api/zerosum/analytics/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_sets" in data
        assert "total_operations" in data
        assert "average_execution_time" in data
        assert "most_used_methods" in data
        assert "convergence_statistics" in data
    
    def test_get_performance_metrics(self, test_client):
        """Test getting performance metrics."""
        response = test_client.get("/api/zerosum/analytics/performance")
        
        assert response.status_code == 200
        data = response.json()
        assert "cache_statistics" in data
        assert "parallel_efficiency" in data
        assert "memory_usage" in data
        assert "operation_throughput" in data
    
    def test_get_convergence_report(self, test_client):
        """Test getting convergence report."""
        response = test_client.get("/api/zerosum/analytics/convergence")
        
        assert response.status_code == 200
        data = response.json()
        assert "convergent_series_count" in data
        assert "divergent_series_count" in data
        assert "convergence_methods_effectiveness" in data
        assert "average_convergence_rate" in data


class TestCacheEndpoints:
    """Cache endpoint tests."""
    
    def test_get_cache_statistics(self, test_client):
        """Test getting cache statistics."""
        response = test_client.get("/api/zerosum/cache/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data
        assert "total_size" in data
        assert "evictions" in data
    
    def test_clear_cache(self, test_client):
        """Test cache clearing."""
        response = test_client.delete("/api/zerosum/cache")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "cleared_items" in data
    
    def test_cache_configuration(self, test_client):
        """Test getting cache configuration."""
        response = test_client.get("/api/zerosum/cache/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "max_size" in data
        assert "eviction_policy" in data
        assert "ttl" in data
        assert "current_size" in data


class TestErrorHandling:
    """Error handling tests."""
    
    def test_invalid_json_payload(self, test_client):
        """Test handling invalid JSON."""
        response = test_client.post(
            "/api/zerosum/sets",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, test_client):
        """Test handling missing required fields."""
        payload = {}  # Empty payload
        
        response = test_client.post("/api/zerosum/sets", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_invalid_uuid_format(self, test_client):
        """Test handling invalid UUID format."""
        invalid_id = "not-a-uuid"
        
        response = test_client.get(f"/api/zerosum/sets/{invalid_id}")
        
        assert response.status_code == 422
    
    def test_method_not_allowed(self, test_client):
        """Test handling disallowed HTTP methods."""
        response = test_client.patch("/api/zerosum/sets")
        
        assert response.status_code == 405
    
    def test_internal_server_error_handling(self, test_client):
        """Test handling internal server errors."""
        # Simulate internal error through mock
        with patch.object(ZeroSumService, 'create_set') as mock_create:
            mock_create.side_effect = Exception("Internal error")
            
            payload = {
                "elements": [1, 2, 3],
                "metadata": {"name": "error_test"}
            }
            
            response = test_client.post("/api/zerosum/sets", json=payload)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data


class TestAuthentication:
    """Authentication tests (if implemented)."""
    
    @pytest.mark.skip(reason="Authentication not implemented yet")
    def test_protected_endpoint_without_auth(self, test_client):
        """Test accessing protected endpoint without authentication."""
        response = test_client.get("/api/zerosum/admin/statistics")
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Authentication not implemented yet")
    def test_protected_endpoint_with_invalid_token(self, test_client):
        """Test accessing with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = test_client.get("/api/zerosum/admin/statistics", headers=headers)
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Authentication not implemented yet")
    def test_protected_endpoint_with_valid_token(self, test_client):
        """Test accessing with valid token."""
        # Get token
        auth_response = test_client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        token = auth_response.json()["access_token"]
        
        # Use token to access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = test_client.get("/api/zerosum/admin/statistics", headers=headers)
        
        assert response.status_code == 200


class TestWebSocketEndpoints:
    """WebSocket endpoint tests (if implemented)."""
    
    @pytest.mark.skip(reason="WebSocket endpoints not implemented yet")
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            async with client.websocket_connect("/ws/zerosum") as websocket:
                # Send message
                await websocket.send_json({
                    "type": "zero_sum_operation",
                    "data": {
                        "elements": [1, -1, 2, -2],
                        "method": "direct"
                    }
                })
                
                # Receive response
                response = await websocket.receive_json()
                
                assert "type" in response
                assert "result" in response
    
    @pytest.mark.skip(reason="WebSocket endpoints not implemented yet")
    @pytest.mark.asyncio
    async def test_websocket_real_time_updates(self):
        """Test receiving real-time updates via WebSocket."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            async with client.websocket_connect("/ws/zerosum/updates") as websocket:
                # Subscribe to updates
                await websocket.send_json({
                    "type": "subscribe",
                    "topics": ["operations", "statistics"]
                })
                
                # Execute operation via REST API
                await client.post("/api/zerosum/sets", json={
                    "elements": [1, 2, 3],
                    "metadata": {"name": "websocket_test"}
                })
                
                # Should receive notification via WebSocket
                update = await websocket.receive_json()
                
                assert update["type"] == "operation_completed"
                assert "data" in update


class TestRateLimiting:
    """Rate limiting tests (if implemented)."""
    
    @pytest.mark.skip(reason="Rate limiting not implemented yet")
    def test_rate_limit_exceeded(self, test_client):
        """Test exceeding rate limit."""
        # Send many requests in a row
        for i in range(100):
            response = test_client.get("/api/zerosum/sets")
            if response.status_code == 429:
                # Rate limit exceeded
                assert "Retry-After" in response.headers
                break
        else:
            pytest.fail("Rate limit was not triggered")
    
    @pytest.mark.skip(reason="Rate limiting not implemented yet")
    def test_rate_limit_headers(self, test_client):
        """Test rate limiting headers."""
        response = test_client.get("/api/zerosum/sets")
        
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestCORS:
    """CORS (Cross-Origin Resource Sharing) tests."""
    
    def test_cors_preflight_request(self, test_client):
        """Test CORS preflight request."""
        response = test_client.options(
            "/api/zerosum/sets",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
    
    def test_cors_actual_request(self, test_client):
        """Test actual CORS request."""
        response = test_client.get(
            "/api/zerosum/sets",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
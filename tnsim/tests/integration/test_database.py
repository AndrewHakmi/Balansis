"""Integration tests for TNSIM database."""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from tnsim.database.connection import DatabaseConnection
from tnsim.database.repository import InfiniteSetRepository
from tnsim.database.config import DatabaseConfig, get_config
from tnsim.core.sets import ZeroSumInfiniteSet


class TestDatabaseConnection:
    """Database connection tests."""
    
    @pytest.mark.asyncio
    async def test_connection_creation(self, db_config):
        """Test database connection creation."""
        connection = DatabaseConnection(db_config)
        
        async with connection.get_connection() as conn:
            # Check that connection is active
            result = await conn.fetchval("SELECT 1")
            assert result == 1
    
    @pytest.mark.asyncio
    async def test_connection_pool(self, db_config):
        """Test connection pool."""
        connection = DatabaseConnection(db_config)
        
        # Create multiple simultaneous connections
        async def test_query():
            async with connection.get_connection() as conn:
                return await conn.fetchval("SELECT pg_backend_pid()")
        
        # Execute queries in parallel
        tasks = [test_query() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All queries should execute successfully
        assert len(results) == 5
        assert all(isinstance(pid, int) for pid in results)
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""
        # Create configuration with invalid parameters
        invalid_config = DatabaseConfig(
            host="invalid_host",
            port=5432,
            database="invalid_db",
            username="invalid_user",
            password="invalid_password"
        )
        
        connection = DatabaseConnection(invalid_config)
        
        with pytest.raises(Exception):  # Expect connection error
            async with connection.get_connection() as conn:
                await conn.fetchval("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_transaction_handling(self, db_connection):
        """Test transaction handling."""
        async with db_connection.get_connection() as conn:
            async with conn.transaction():
                # Create temporary table for test
                await conn.execute("""
                    CREATE TEMP TABLE test_transaction (
                        id SERIAL PRIMARY KEY,
                        value TEXT
                    )
                """)
                
                # Insert data
                await conn.execute(
                    "INSERT INTO test_transaction (value) VALUES ($1)",
                    "test_value"
                )
                
                # Check that data is inserted
                result = await conn.fetchval(
                    "SELECT value FROM test_transaction WHERE id = 1"
                )
                assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db_connection):
        """Test transaction rollback."""
        async with db_connection.get_connection() as conn:
            # Create temporary table
            await conn.execute("""
                CREATE TEMP TABLE test_rollback (
                    id SERIAL PRIMARY KEY,
                    value TEXT
                )
            """)
            
            try:
                async with conn.transaction():
                    # Insert data
                    await conn.execute(
                        "INSERT INTO test_rollback (value) VALUES ($1)",
                        "test_value"
                    )
                    
                    # Force an error
                    raise Exception("Test rollback")
            except Exception:
                pass  # Expected error
            
            # Check that data was not saved
            count = await conn.fetchval("SELECT COUNT(*) FROM test_rollback")
            assert count == 0


class TestInfiniteSetRepository:
    """Infinite set repository tests."""
    
    @pytest.mark.asyncio
    async def test_create_infinite_set(self, infinite_set_repository, sample_harmonic_set):
        """Test infinite set creation."""
        set_id = await infinite_set_repository.create_set(sample_harmonic_set)
        
        assert isinstance(set_id, uuid.UUID)
        
        # Check that set is created in database
        retrieved_set = await infinite_set_repository.get_set(set_id)
        assert retrieved_set is not None
        assert retrieved_set.metadata.get("name") == sample_harmonic_set.metadata.get("name")
    
    @pytest.mark.asyncio
    async def test_get_infinite_set(self, infinite_set_repository, sample_set_id):
        """Test infinite set retrieval."""
        retrieved_set = await infinite_set_repository.get_set(sample_set_id)
        
        assert retrieved_set is not None
        assert isinstance(retrieved_set, ZeroSumInfiniteSet)
        assert len(retrieved_set.elements) > 0
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_set(self, infinite_set_repository):
        """Test retrieval of non-existent set."""
        fake_id = uuid.uuid4()
        retrieved_set = await infinite_set_repository.get_set(fake_id)
        
        assert retrieved_set is None
    
    @pytest.mark.asyncio
    async def test_update_infinite_set(self, infinite_set_repository, sample_set_id):
        """Test infinite set update."""
        # Get original set
        original_set = await infinite_set_repository.get_set(sample_set_id)
        
        # Update metadata
        original_set.metadata["updated"] = True
        original_set.metadata["update_time"] = datetime.now().isoformat()
        
        # Save changes
        success = await infinite_set_repository.update_set(sample_set_id, original_set)
        assert success
        
        # Check that changes are saved
        updated_set = await infinite_set_repository.get_set(sample_set_id)
        assert updated_set.metadata.get("updated") is True
        assert "update_time" in updated_set.metadata
    
    @pytest.mark.asyncio
    async def test_delete_infinite_set(self, infinite_set_repository, sample_alternating_set):
        """Test infinite set deletion."""
        # Create set for deletion
        set_id = await infinite_set_repository.create_set(sample_alternating_set)
        
        # Check that set exists
        retrieved_set = await infinite_set_repository.get_set(set_id)
        assert retrieved_set is not None
        
        # Delete set
        success = await infinite_set_repository.delete_set(set_id)
        assert success
        
        # Check that set is deleted
        deleted_set = await infinite_set_repository.get_set(set_id)
        assert deleted_set is None
    
    @pytest.mark.asyncio
    async def test_list_infinite_sets(self, infinite_set_repository):
        """Test infinite sets listing."""
        sets, total = await infinite_set_repository.list_sets(page=1, page_size=10)
        
        assert isinstance(sets, list)
        assert isinstance(total, int)
        assert len(sets) <= 10
        assert total >= 0
    
    @pytest.mark.asyncio
    async def test_list_sets_with_filters(self, infinite_set_repository):
        """Test listing sets with filters."""
        # Create set with specific metadata
        test_set = ZeroSumInfiniteSet(
            elements=[1, -1, 2, -2],
            metadata={
                "name": "filter_test",
                "series_type": "custom",
                "created_by": "test_user"
            }
        )
        
        set_id = await infinite_set_repository.create_set(test_set)
        
        # Filter by name
        sets, total = await infinite_set_repository.list_sets(
            page=1, page_size=10,
            filters={"name": "filter_test"}
        )
        
        assert total >= 1
        assert any(s.metadata.get("name") == "filter_test" for s in sets)
    
    @pytest.mark.asyncio
    async def test_search_sets(self, infinite_set_repository):
        """Test set search."""
        # Create set with searchable metadata
        searchable_set = ZeroSumInfiniteSet(
            elements=[3, -3, 4, -4],
            metadata={
                "name": "searchable_harmonic",
                "description": "A harmonic series for search testing",
                "tags": ["harmonic", "convergent", "test"]
            }
        )
        
        set_id = await infinite_set_repository.create_set(searchable_set)
        
        # Search by keyword
        results = await infinite_set_repository.search_sets("harmonic")
        
        assert len(results) >= 1
        assert any(r.metadata.get("name") == "searchable_harmonic" for r in results)
    
    @pytest.mark.asyncio
    async def test_get_set_statistics(self, infinite_set_repository):
        """Test set statistics retrieval."""
        stats = await infinite_set_repository.get_statistics()
        
        assert "total_sets" in stats
        assert "series_types" in stats
        assert "average_elements_count" in stats
        assert "creation_dates" in stats
        
        assert isinstance(stats["total_sets"], int)
        assert stats["total_sets"] >= 0


class TestOperationLogging:
    """Operation logging tests."""
    
    @pytest.mark.asyncio
    async def test_log_zero_sum_operation(self, infinite_set_repository, sample_set_id):
        """Test zero sum operation logging."""
        operation_data = {
            "operation_type": "zero_sum",
            "method": "direct",
            "result": 0.0,
            "execution_time": 0.001,
            "parameters": {"tolerance": 1e-12}
        }
        
        log_id = await infinite_set_repository.log_operation(
            sample_set_id, operation_data
        )
        
        assert isinstance(log_id, uuid.UUID)
        
        # Check that operation is logged
        logs = await infinite_set_repository.get_operation_logs(sample_set_id)
        assert len(logs) >= 1
        assert any(log["operation_type"] == "zero_sum" for log in logs)
    
    @pytest.mark.asyncio
    async def test_get_operation_logs(self, infinite_set_repository, sample_set_id):
        """Test operation logs retrieval."""
        # Log multiple operations
        operations = [
            {
                "operation_type": "zero_sum",
                "method": "direct",
                "result": 0.0,
                "execution_time": 0.001
            },
            {
                "operation_type": "validation",
                "method": "basic",
                "result": True,
                "execution_time": 0.0005
            },
            {
                "operation_type": "convergence_analysis",
                "method": "ratio_test",
                "result": "convergent",
                "execution_time": 0.002
            }
        ]
        
        for op in operations:
            await infinite_set_repository.log_operation(sample_set_id, op)
        
        # Get logs
        logs = await infinite_set_repository.get_operation_logs(sample_set_id)
        
        assert len(logs) >= 3
        operation_types = {log["operation_type"] for log in logs}
        assert "zero_sum" in operation_types
        assert "validation" in operation_types
        assert "convergence_analysis" in operation_types
    
    @pytest.mark.asyncio
    async def test_get_operation_statistics(self, infinite_set_repository):
        """Test getting operation statistics."""
        stats = await infinite_set_repository.get_operation_statistics()
        
        assert "total_operations" in stats
        assert "operations_by_type" in stats
        assert "average_execution_time" in stats
        assert "most_used_methods" in stats
        
        assert isinstance(stats["total_operations"], int)
        assert stats["total_operations"] >= 0


class TestCompensationPairs:
    """Tests for working with compensation pairs."""
    
    @pytest.mark.asyncio
    async def test_store_compensation_pair(self, infinite_set_repository, sample_set_id):
        """Test storing a compensation pair."""
        compensation_data = {
            "original_element": 1.5,
            "compensating_element": -1.5,
            "compensation_quality": 0.99,
            "method": "iterative",
            "iterations": 10
        }
        
        pair_id = await infinite_set_repository.store_compensation_pair(
            sample_set_id, compensation_data
        )
        
        assert isinstance(pair_id, uuid.UUID)
    
    @pytest.mark.asyncio
    async def test_get_compensation_pairs(self, infinite_set_repository, sample_set_id):
        """Test getting compensation pairs."""
        # Store several pairs
        pairs_data = [
            {
                "original_element": 2.0,
                "compensating_element": -2.0,
                "compensation_quality": 1.0,
                "method": "direct"
            },
            {
                "original_element": 3.5,
                "compensating_element": -3.5,
                "compensation_quality": 0.98,
                "method": "adaptive"
            }
        ]
        
        for pair_data in pairs_data:
            await infinite_set_repository.store_compensation_pair(
                sample_set_id, pair_data
            )
        
        # Get pairs
        pairs = await infinite_set_repository.get_compensation_pairs(sample_set_id)
        
        assert len(pairs) >= 2
        qualities = [pair["compensation_quality"] for pair in pairs]
        assert 1.0 in qualities
        assert 0.98 in qualities
    
    @pytest.mark.asyncio
    async def test_find_best_compensation_pairs(self, infinite_set_repository):
        """Test finding the best compensation pairs."""
        best_pairs = await infinite_set_repository.get_best_compensation_pairs(
            limit=5, min_quality=0.95
        )
        
        assert isinstance(best_pairs, list)
        assert len(best_pairs) <= 5
        
        # Check that all pairs have quality >= 0.95
        for pair in best_pairs:
            assert pair["compensation_quality"] >= 0.95
        
        # Check that pairs are sorted by quality (descending)
        if len(best_pairs) > 1:
            qualities = [pair["compensation_quality"] for pair in best_pairs]
            assert qualities == sorted(qualities, reverse=True)


class TestDatabaseMigrations:
    """Tests for database migrations."""
    
    @pytest.mark.asyncio
    async def test_check_tables_exist(self, db_connection):
        """Test checking table existence."""
        expected_tables = [
            "infinite_sets",
            "set_elements", 
            "compensation_pairs",
            "operation_logs"
        ]
        
        async with db_connection.get_connection() as conn:
            for table in expected_tables:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = $1
                    )
                """, table)
                
                assert exists, f"Table {table} does not exist"
    
    @pytest.mark.asyncio
    async def test_check_indexes_exist(self, db_connection):
        """Test checking index existence."""
        expected_indexes = [
            "idx_infinite_sets_created_at",
            "idx_set_elements_set_id",
            "idx_compensation_pairs_set_id",
            "idx_operation_logs_set_id",
            "idx_operation_logs_created_at"
        ]
        
        async with db_connection.get_connection() as conn:
            for index in expected_indexes:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM pg_indexes 
                        WHERE schemaname = 'public' 
                        AND indexname = $1
                    )
                """, index)
                
                assert exists, f"Index {index} does not exist"
    
    @pytest.mark.asyncio
    async def test_check_functions_exist(self, db_connection):
        """Test checking function existence."""
        async with db_connection.get_connection() as conn:
            # Check the updated_at update function
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_proc p
                    JOIN pg_namespace n ON p.pronamespace = n.oid
                    WHERE n.nspname = 'public'
                    AND p.proname = 'update_updated_at_column'
                )
            """)
            
            assert exists, "Function update_updated_at_column does not exist"
    
    @pytest.mark.asyncio
    async def test_check_triggers_exist(self, db_connection):
        """Test checking trigger existence."""
        expected_triggers = [
            ("infinite_sets", "update_infinite_sets_updated_at"),
            ("compensation_pairs", "update_compensation_pairs_updated_at"),
            ("operation_logs", "update_operation_logs_updated_at")
        ]
        
        async with db_connection.get_connection() as conn:
            for table, trigger in expected_triggers:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM pg_trigger t
                        JOIN pg_class c ON t.tgrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = 'public'
                        AND c.relname = $1
                        AND t.tgname = $2
                    )
                """, table, trigger)
                
                assert exists, f"Trigger {trigger} on table {table} does not exist"


class TestDatabasePerformance:
    """Database performance tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_bulk_insert_performance(self, infinite_set_repository):
        """Test bulk insert performance."""
        import time
        
        # Create sets for bulk insertion
        sets_to_insert = []
        for i in range(100):
            test_set = ZeroSumInfiniteSet(
                elements=[i, -i, i*0.5, -i*0.5],
                metadata={
                    "name": f"bulk_test_{i}",
                    "series_type": "custom",
                    "batch_id": "performance_test"
                }
            )
            sets_to_insert.append(test_set)
        
        # Measure insertion time
        start_time = time.time()
        
        # Insert sets (sequentially for simplicity)
        inserted_ids = []
        for test_set in sets_to_insert:
            set_id = await infinite_set_repository.create_set(test_set)
            inserted_ids.append(set_id)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Check that all sets are inserted
        assert len(inserted_ids) == 100
        
        # Check performance (should be faster than 10 seconds)
        assert execution_time < 10.0, f"Bulk insert took too long: {execution_time}s"
        
        print(f"Bulk insert of 100 sets took: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_query_performance(self, infinite_set_repository):
        """Test query performance."""
        import time
        
        # Measure execution time of various queries
        start_time = time.time()
        
        # List sets
        sets, total = await infinite_set_repository.list_sets(page=1, page_size=50)
        
        list_time = time.time() - start_time
        
        # Search
        start_time = time.time()
        search_results = await infinite_set_repository.search_sets("test")
        search_time = time.time() - start_time
        
        # Statistics
        start_time = time.time()
        stats = await infinite_set_repository.get_statistics()
        stats_time = time.time() - start_time
        
        # Check performance
        assert list_time < 1.0, f"List query took too long: {list_time}s"
        assert search_time < 2.0, f"Search query took too long: {search_time}s"
        assert stats_time < 1.0, f"Statistics query took too long: {stats_time}s"
        
        print(f"Query performance - List: {list_time:.3f}s, Search: {search_time:.3f}s, Stats: {stats_time:.3f}s")


class TestDatabaseConcurrency:
    """Database concurrent access tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, infinite_set_repository):
        """Test concurrent inserts."""
        async def create_test_set(index):
            test_set = ZeroSumInfiniteSet(
                elements=[index, -index],
                metadata={
                    "name": f"concurrent_test_{index}",
                    "thread_id": index
                }
            )
            return await infinite_set_repository.create_set(test_set)
        
        # Create sets concurrently
        tasks = [create_test_set(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all operations completed successfully
        successful_results = [r for r in results if isinstance(r, uuid.UUID)]
        assert len(successful_results) == 10
        
        # Check that all IDs are unique
        assert len(set(successful_results)) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_reads_writes(self, infinite_set_repository, sample_set_id):
        """Test concurrent reads and writes."""
        async def read_set():
            return await infinite_set_repository.get_set(sample_set_id)
        
        async def log_operation(index):
            operation_data = {
                "operation_type": "test_concurrent",
                "method": "concurrent",
                "result": index,
                "execution_time": 0.001
            }
            return await infinite_set_repository.log_operation(sample_set_id, operation_data)
        
        # Mix read and write operations
        tasks = []
        for i in range(5):
            tasks.append(read_set())
            tasks.append(log_operation(i))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all operations completed without errors
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Found errors in concurrent operations: {errors}"


class TestDatabaseBackup:
    """Backup tests (if implemented)."""
    
    @pytest.mark.skip(reason="Backup functionality not implemented yet")
    @pytest.mark.asyncio
    async def test_create_backup(self, infinite_set_repository):
        """Test creating a backup."""
        backup_data = await infinite_set_repository.create_backup()
        
        assert "infinite_sets" in backup_data
        assert "set_elements" in backup_data
        assert "compensation_pairs" in backup_data
        assert "operation_logs" in backup_data
        assert "metadata" in backup_data
        
        # Check backup metadata
        metadata = backup_data["metadata"]
        assert "created_at" in metadata
        assert "version" in metadata
        assert "total_records" in metadata
    
    @pytest.mark.skip(reason="Backup functionality not implemented yet")
    @pytest.mark.asyncio
    async def test_restore_backup(self, infinite_set_repository):
        """Test restoring from backup."""
        # Create backup
        backup_data = await infinite_set_repository.create_backup()
        
        # Clear database
        await infinite_set_repository.clear_all_data()
        
        # Restore from backup
        success = await infinite_set_repository.restore_backup(backup_data)
        assert success
        
        # Check that data is restored
        stats = await infinite_set_repository.get_statistics()
        assert stats["total_sets"] > 0
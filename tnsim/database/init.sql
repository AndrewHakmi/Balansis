-- TNSIM Database Initialization Script
-- Theory of Null-Sum Infinite Multitudes (TNSIM)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE compensation_method AS ENUM (
    'direct',
    'compensated', 
    'iterative',
    'adaptive',
    'stabilized'
);

CREATE TYPE convergence_test AS ENUM (
    'ratio',
    'root', 
    'integral',
    'comparison'
);

CREATE TYPE set_status AS ENUM (
    'active',
    'archived',
    'processing',
    'error'
);

-- Main table for infinite sets
CREATE TABLE infinite_sets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status set_status DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    
    -- Zero-sum properties
    is_zero_sum BOOLEAN DEFAULT FALSE,
    compensation_value DOUBLE PRECISION,
    compensation_method compensation_method DEFAULT 'direct',
    
    -- Convergence properties
    is_convergent BOOLEAN,
    convergence_test convergence_test,
    convergence_rate DOUBLE PRECISION,
    
    -- Performance metrics
    element_count BIGINT DEFAULT 0,
    last_operation_time INTERVAL,
    cache_hits BIGINT DEFAULT 0,
    cache_misses BIGINT DEFAULT 0,
    
    -- Indexing
    search_vector tsvector,
    
    CONSTRAINT positive_element_count CHECK (element_count >= 0),
    CONSTRAINT valid_compensation_value CHECK (compensation_value IS NULL OR abs(compensation_value) < 1e308)
);

-- Table for set elements
CREATE TABLE set_elements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    set_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    element_index BIGINT NOT NULL,
    element_value DOUBLE PRECISION NOT NULL,
    is_compensating BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional properties
    precision_bits INTEGER DEFAULT 64,
    error_estimate DOUBLE PRECISION,
    
    UNIQUE(set_id, element_index),
    CONSTRAINT valid_element_value CHECK (abs(element_value) < 1e308),
    CONSTRAINT valid_precision CHECK (precision_bits > 0 AND precision_bits <= 128)
);

-- Table for compensation pairs
CREATE TABLE compensation_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    set_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    primary_element_id UUID NOT NULL REFERENCES set_elements(id) ON DELETE CASCADE,
    compensating_element_id UUID NOT NULL REFERENCES set_elements(id) ON DELETE CASCADE,
    compensation_quality DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Balansis integration
    balansis_method VARCHAR(50),
    balansis_precision INTEGER DEFAULT 64,
    
    UNIQUE(primary_element_id, compensating_element_id),
    CONSTRAINT different_elements CHECK (primary_element_id != compensating_element_id),
    CONSTRAINT valid_quality CHECK (compensation_quality >= 0.0 AND compensation_quality <= 1.0)
);

-- Table for operation results
CREATE TABLE operation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    set_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    operation_type VARCHAR(50) NOT NULL,
    method compensation_method NOT NULL,
    result_value DOUBLE PRECISION,
    error_estimate DOUBLE PRECISION,
    execution_time INTERVAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Performance metrics
    iterations_count INTEGER,
    memory_usage_mb DOUBLE PRECISION,
    cpu_usage_percent DOUBLE PRECISION,
    
    -- Parallel processing
    worker_count INTEGER DEFAULT 1,
    parallel_efficiency DOUBLE PRECISION,
    
    CONSTRAINT positive_iterations CHECK (iterations_count IS NULL OR iterations_count > 0),
    CONSTRAINT valid_worker_count CHECK (worker_count > 0)
);

-- Table for convergence analysis
CREATE TABLE convergence_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    set_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    test_type convergence_test NOT NULL,
    is_convergent BOOLEAN NOT NULL,
    convergence_rate DOUBLE PRECISION,
    radius_of_convergence DOUBLE PRECISION,
    partial_sums DOUBLE PRECISION[],
    test_values DOUBLE PRECISION[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Analysis parameters
    max_terms INTEGER DEFAULT 1000,
    tolerance DOUBLE PRECISION DEFAULT 1e-15,
    
    CONSTRAINT positive_max_terms CHECK (max_terms > 0),
    CONSTRAINT positive_tolerance CHECK (tolerance > 0)
);

-- Table for cache entries
CREATE TABLE cache_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) NOT NULL UNIQUE,
    cache_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    access_count BIGINT DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Cache statistics
    size_bytes INTEGER,
    compression_ratio DOUBLE PRECISION DEFAULT 1.0,
    
    CONSTRAINT positive_access_count CHECK (access_count >= 0),
    CONSTRAINT valid_expiration CHECK (expires_at IS NULL OR expires_at > created_at)
);

-- Table for attention mechanisms (Zero-Sum Attention)
CREATE TABLE attention_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    model_dim INTEGER NOT NULL,
    num_heads INTEGER NOT NULL,
    dropout_rate DOUBLE PRECISION DEFAULT 0.0,
    use_zero_sum BOOLEAN DEFAULT TRUE,
    compensation_method compensation_method DEFAULT 'adaptive',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Performance parameters
    batch_size INTEGER DEFAULT 32,
    sequence_length INTEGER DEFAULT 512,
    
    CONSTRAINT positive_model_dim CHECK (model_dim > 0),
    CONSTRAINT positive_num_heads CHECK (num_heads > 0),
    CONSTRAINT valid_dropout CHECK (dropout_rate >= 0.0 AND dropout_rate <= 1.0),
    CONSTRAINT valid_batch_size CHECK (batch_size > 0),
    CONSTRAINT valid_sequence_length CHECK (sequence_length > 0)
);

-- Create indexes for performance
CREATE INDEX idx_infinite_sets_status ON infinite_sets(status);
CREATE INDEX idx_infinite_sets_created_at ON infinite_sets(created_at);
CREATE INDEX idx_infinite_sets_is_zero_sum ON infinite_sets(is_zero_sum);
CREATE INDEX idx_infinite_sets_search_vector ON infinite_sets USING GIN(search_vector);
CREATE INDEX idx_infinite_sets_metadata ON infinite_sets USING GIN(metadata);

CREATE INDEX idx_set_elements_set_id ON set_elements(set_id);
CREATE INDEX idx_set_elements_index ON set_elements(element_index);
CREATE INDEX idx_set_elements_value ON set_elements(element_value);
CREATE INDEX idx_set_elements_compensating ON set_elements(is_compensating);

CREATE INDEX idx_compensation_pairs_set_id ON compensation_pairs(set_id);
CREATE INDEX idx_compensation_pairs_quality ON compensation_pairs(compensation_quality);

CREATE INDEX idx_operation_results_set_id ON operation_results(set_id);
CREATE INDEX idx_operation_results_created_at ON operation_results(created_at);
CREATE INDEX idx_operation_results_method ON operation_results(method);

CREATE INDEX idx_convergence_analysis_set_id ON convergence_analysis(set_id);
CREATE INDEX idx_convergence_analysis_test_type ON convergence_analysis(test_type);

CREATE INDEX idx_cache_entries_key ON cache_entries(cache_key);
CREATE INDEX idx_cache_entries_expires_at ON cache_entries(expires_at);
CREATE INDEX idx_cache_entries_last_accessed ON cache_entries(last_accessed);

CREATE INDEX idx_attention_configs_name ON attention_configs(name);

-- Create triggers for automatic updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_infinite_sets_updated_at 
    BEFORE UPDATE ON infinite_sets 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update search vector
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', 
        COALESCE(NEW.name, '') || ' ' || 
        COALESCE(NEW.description, '')
    );
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_infinite_sets_search_vector
    BEFORE INSERT OR UPDATE ON infinite_sets
    FOR EACH ROW EXECUTE FUNCTION update_search_vector();

-- Function to update element count
CREATE OR REPLACE FUNCTION update_element_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE infinite_sets 
        SET element_count = element_count + 1
        WHERE id = NEW.set_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE infinite_sets 
        SET element_count = element_count - 1
        WHERE id = OLD.set_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_set_element_count
    AFTER INSERT OR DELETE ON set_elements
    FOR EACH ROW EXECUTE FUNCTION update_element_count();

-- Function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM cache_entries 
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Create views for common queries
CREATE VIEW active_zero_sum_sets AS
SELECT 
    id,
    name,
    description,
    compensation_value,
    compensation_method,
    element_count,
    created_at
FROM infinite_sets
WHERE status = 'active' AND is_zero_sum = TRUE;

CREATE VIEW convergent_sets AS
SELECT 
    s.id,
    s.name,
    s.is_convergent,
    s.convergence_test,
    s.convergence_rate,
    ca.radius_of_convergence
FROM infinite_sets s
LEFT JOIN convergence_analysis ca ON s.id = ca.set_id
WHERE s.is_convergent = TRUE;

CREATE VIEW performance_stats AS
SELECT 
    s.id,
    s.name,
    s.element_count,
    AVG(or_table.execution_time) as avg_execution_time,
    AVG(or_table.memory_usage_mb) as avg_memory_usage,
    SUM(s.cache_hits) as total_cache_hits,
    SUM(s.cache_misses) as total_cache_misses,
    CASE 
        WHEN SUM(s.cache_hits + s.cache_misses) > 0 
        THEN SUM(s.cache_hits)::DOUBLE PRECISION / SUM(s.cache_hits + s.cache_misses)
        ELSE 0
    END as cache_hit_ratio
FROM infinite_sets s
LEFT JOIN operation_results or_table ON s.id = or_table.set_id
GROUP BY s.id, s.name, s.element_count;

-- Insert sample data for testing
INSERT INTO infinite_sets (name, description, is_zero_sum, compensation_method) VALUES
('Harmonic Series', 'Classic divergent harmonic series 1/n', FALSE, 'compensated'),
('Alternating Harmonic', 'Convergent alternating harmonic series (-1)^n/n', TRUE, 'iterative'),
('Geometric Series', 'Convergent geometric series with ratio 1/2', TRUE, 'direct'),
('Riemann Zeta', 'Riemann zeta function series for s=2', TRUE, 'adaptive');

-- Insert sample attention configuration
INSERT INTO attention_configs (name, model_dim, num_heads, use_zero_sum, compensation_method) VALUES
('BERT-Base-ZeroSum', 768, 12, TRUE, 'adaptive'),
('GPT-2-ZeroSum', 1024, 16, TRUE, 'iterative'),
('Transformer-Small', 512, 8, TRUE, 'compensated');

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO tnsim_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO tnsim_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO tnsim_user;

-- Create indexes on foreign keys for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_set_elements_set_id_fk ON set_elements(set_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compensation_pairs_set_id_fk ON compensation_pairs(set_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_operation_results_set_id_fk ON operation_results(set_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_convergence_analysis_set_id_fk ON convergence_analysis(set_id);

-- Analyze tables for better query planning
ANALYZE;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'TNSIM database initialized successfully at %', CURRENT_TIMESTAMP;
END
$$;
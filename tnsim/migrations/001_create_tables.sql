-- Migration for creating TNSIM (Zero Sum Theory of Infinite Sets) tables
-- Version: 001
-- Date: 2024-01-20

-- Creating extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Table for infinite sets
CREATE TABLE IF NOT EXISTS infinite_sets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    series_type VARCHAR(50) NOT NULL CHECK (series_type IN ('harmonic', 'alternating', 'geometric', 'custom')),
    parameters JSONB NOT NULL DEFAULT '{}',
    description TEXT,
    convergence_info JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for infinite_sets table
CREATE INDEX IF NOT EXISTS idx_infinite_sets_series_type ON infinite_sets(series_type);
CREATE INDEX IF NOT EXISTS idx_infinite_sets_created_at ON infinite_sets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_infinite_sets_name_gin ON infinite_sets USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_infinite_sets_parameters_gin ON infinite_sets USING gin(parameters);

-- Table for set elements
CREATE TABLE IF NOT EXISTS set_elements (
    id BIGSERIAL PRIMARY KEY,
    set_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(set_id, position)
);

-- Indexes for set_elements table
CREATE INDEX IF NOT EXISTS idx_set_elements_set_id ON set_elements(set_id);
CREATE INDEX IF NOT EXISTS idx_set_elements_position ON set_elements(set_id, position);
CREATE INDEX IF NOT EXISTS idx_set_elements_value ON set_elements(value);

-- Table for compensation pairs
CREATE TABLE IF NOT EXISTS compensation_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    set1_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    set2_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    compensation_quality DOUBLE PRECISION NOT NULL CHECK (compensation_quality >= 0 AND compensation_quality <= 1),
    method_used VARCHAR(50) NOT NULL,
    tolerance DOUBLE PRECISION NOT NULL CHECK (tolerance > 0),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(set1_id, set2_id)
);

-- Indexes for compensation_pairs table
CREATE INDEX IF NOT EXISTS idx_compensation_pairs_set1_id ON compensation_pairs(set1_id);
CREATE INDEX IF NOT EXISTS idx_compensation_pairs_set2_id ON compensation_pairs(set2_id);
CREATE INDEX IF NOT EXISTS idx_compensation_pairs_quality ON compensation_pairs(compensation_quality DESC);
CREATE INDEX IF NOT EXISTS idx_compensation_pairs_method ON compensation_pairs(method_used);

-- Table for operation logs
CREATE TABLE IF NOT EXISTS operation_logs (
    id BIGSERIAL PRIMARY KEY,
    operation_id UUID NOT NULL,
    operation_type VARCHAR(50) NOT NULL,
    parameters JSONB DEFAULT '{}',
    result JSONB,
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failed', 'partial', 'timeout')),
    execution_time DOUBLE PRECISION NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for operation_logs table
CREATE INDEX IF NOT EXISTS idx_operation_logs_operation_id ON operation_logs(operation_id);
CREATE INDEX IF NOT EXISTS idx_operation_logs_type ON operation_logs(operation_type);
CREATE INDEX IF NOT EXISTS idx_operation_logs_status ON operation_logs(status);
CREATE INDEX IF NOT EXISTS idx_operation_logs_created_at ON operation_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_operation_logs_execution_time ON operation_logs(execution_time DESC);

-- Function for automatic updated_at column update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for automatic updated_at update in infinite_sets
CREATE TRIGGER update_infinite_sets_updated_at
    BEFORE UPDATE ON infinite_sets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Inserting initial data

-- Example harmonic series
INSERT INTO infinite_sets (id, name, series_type, parameters, description, convergence_info)
VALUES (
    uuid_generate_v4(),
    'Classical harmonic series',
    'harmonic',
    '{"p": 1.0}',
    'Classical harmonic series 1/n, which diverges',
    '{"is_convergent": false, "is_divergent": true, "convergence_rate": null, "sum_estimate": null}'
) ON CONFLICT DO NOTHING;

-- Example p-harmonic series (convergent)
INSERT INTO infinite_sets (id, name, series_type, parameters, description, convergence_info)
VALUES (
    uuid_generate_v4(),
    'P-harmonic series (p=2)',
    'harmonic',
    '{"p": 2.0}',
    'P-harmonic series 1/n^2, which converges to π²/6',
    '{"is_convergent": true, "is_divergent": false, "convergence_rate": 0.95, "sum_estimate": 1.6449340668}'
) ON CONFLICT DO NOTHING;

-- Example alternating series
INSERT INTO infinite_sets (id, name, series_type, parameters, description, convergence_info)
VALUES (
    uuid_generate_v4(),
    'Alternating harmonic series',
    'alternating',
    '{"p": 1.0}',
    'Alternating harmonic series (-1)^(n+1)/n, which converges to ln(2)',
    '{"is_convergent": true, "is_absolutely_convergent": false, "convergence_rate": 0.85, "sum_estimate": 0.6931471806}'
) ON CONFLICT DO NOTHING;

-- Example geometric series (convergent)
INSERT INTO infinite_sets (id, name, series_type, parameters, description, convergence_info)
VALUES (
    uuid_generate_v4(),
    'Geometric series (r=0.5)',
    'geometric',
    '{"ratio": 0.5}',
    'Geometric series with ratio 0.5, which converges to 2',
    '{"is_convergent": true, "is_absolutely_convergent": true, "convergence_rate": 0.99, "sum_estimate": 2.0}'
) ON CONFLICT DO NOTHING;

-- Example geometric series (divergent)
INSERT INTO infinite_sets (id, name, series_type, parameters, description, convergence_info)
VALUES (
    uuid_generate_v4(),
    'Geometric series (r=1.5)',
    'geometric',
    '{"ratio": 1.5}',
    'Geometric series with ratio 1.5, which diverges',
    '{"is_convergent": false, "is_divergent": true, "convergence_rate": null, "sum_estimate": null}'
) ON CONFLICT DO NOTHING;

-- Creating views for analytics

-- View for series type statistics
CREATE OR REPLACE VIEW series_type_stats AS
SELECT 
    series_type,
    COUNT(*) as total_count,
    COUNT(CASE WHEN convergence_info->>'is_convergent' = 'true' THEN 1 END) as convergent_count,
    COUNT(CASE WHEN convergence_info->>'is_divergent' = 'true' THEN 1 END) as divergent_count,
    AVG(CASE WHEN convergence_info->>'convergence_rate' IS NOT NULL 
        THEN (convergence_info->>'convergence_rate')::DOUBLE PRECISION END) as avg_convergence_rate
FROM infinite_sets
GROUP BY series_type;

-- View for operation statistics
CREATE OR REPLACE VIEW operation_stats AS
SELECT 
    operation_type,
    status,
    COUNT(*) as operation_count,
    AVG(execution_time) as avg_execution_time,
    MIN(execution_time) as min_execution_time,
    MAX(execution_time) as max_execution_time,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY execution_time) as median_execution_time
FROM operation_logs
GROUP BY operation_type, status
ORDER BY operation_type, status;

-- View for best compensation pairs
CREATE OR REPLACE VIEW best_compensation_pairs AS
SELECT 
    cp.id,
    s1.name as set1_name,
    s2.name as set2_name,
    cp.compensation_quality,
    cp.method_used,
    cp.tolerance,
    cp.created_at
FROM compensation_pairs cp
JOIN infinite_sets s1 ON cp.set1_id = s1.id
JOIN infinite_sets s2 ON cp.set2_id = s2.id
WHERE cp.compensation_quality > 0.9
ORDER BY cp.compensation_quality DESC, cp.created_at DESC;

-- Table comments
COMMENT ON TABLE infinite_sets IS 'Table for storing definitions of infinite sets and their properties';
COMMENT ON TABLE set_elements IS 'Table for storing computed elements of infinite sets';
COMMENT ON TABLE compensation_pairs IS 'Table for storing pairs of sets that compensate each other';
COMMENT ON TABLE operation_logs IS 'Table for logging all operations with sets';

-- Column comments
COMMENT ON COLUMN infinite_sets.series_type IS 'Series type: harmonic, alternating, geometric, custom';
COMMENT ON COLUMN infinite_sets.parameters IS 'JSON with series parameters (e.g., p for harmonic series)';
COMMENT ON COLUMN infinite_sets.convergence_info IS 'JSON with series convergence information';
COMMENT ON COLUMN compensation_pairs.compensation_quality IS 'Compensation quality from 0 to 1, where 1 is perfect compensation';
COMMENT ON COLUMN operation_logs.execution_time IS 'Operation execution time in seconds';

-- Creating user and setting up access rights (optional)
-- Uncomment the following lines if needed

-- CREATE USER tnsim_user WITH PASSWORD 'tnsim_password';
-- GRANT CONNECT ON DATABASE tnsim TO tnsim_user;
-- GRANT USAGE ON SCHEMA public TO tnsim_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO tnsim_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO tnsim_user;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO tnsim_user;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO tnsim_user;

COMMIT;
-- Postgres 15.4

create function PreserveMarks() returns trigger as
$$
begin
	if old.Mark > new.Mark then
		return old;
	end if;
	return new;
end;
$$ language plpgsql;

create trigger PreserveMarks before update on Marks
for each row execute function PreserveMarks();
